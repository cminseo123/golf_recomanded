"""
Generate weak-label training data for the FITTED golf fitting SLM.

This script is the practical bridge between `DATA_RESEARCH.md` and QLoRA fine-tuning:

1. It uses research-grounded fitting heuristics for launch/spin windows.
2. It builds structured golfer profiles and swing data.
3. It produces rule-based recommendations plus Korean natural-language explanations.
4. It writes chat-format JSONL ready for Qwen2.5 SFT / QLoRA.

Output:
    data/golf_fitting_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple


SYSTEM_PROMPT = (
    "당신은 20년 경력의 전문 골프 클럽 피터입니다. "
    "사용자의 체형, 스윙 데이터, 구질, 고민 사항을 함께 보고 "
    "드라이버, 아이언, 웨지, 퍼터 피팅 조언을 해주세요. "
    "답변은 한국어로 작성하고, 추천 이유와 시타 포인트를 구체적으로 설명하세요."
)


SPEED_WINDOWS = [
    {
        "label": "slow",
        "min_speed": 60,
        "max_speed": 84,
        "launch_range": (15.0, 18.0),
        "spin_range": (2500, 3200),
        "attack_range": (1.0, 4.0),
    },
    {
        "label": "mid",
        "min_speed": 85,
        "max_speed": 95,
        "launch_range": (14.0, 17.0),
        "spin_range": (2200, 2800),
        "attack_range": (3.0, 5.0),
    },
    {
        "label": "mid_fast",
        "min_speed": 96,
        "max_speed": 105,
        "launch_range": (13.0, 15.0),
        "spin_range": (2000, 2500),
        "attack_range": (3.0, 5.0),
    },
    {
        "label": "fast",
        "min_speed": 106,
        "max_speed": 130,
        "launch_range": (11.0, 14.0),
        "spin_range": (1800, 2200),
        "attack_range": (4.0, 6.0),
    },
]


SHAFT_CATALOG = [
    {
        "name": "Fujikura Ventus Red",
        "speed_range": (70, 90),
        "launch_bias": "high",
        "spin_bias": "high",
    },
    {
        "name": "Fujikura Ventus Blue",
        "speed_range": (85, 105),
        "launch_bias": "mid",
        "spin_bias": "mid",
    },
    {
        "name": "Fujikura Ventus Black",
        "speed_range": (100, 125),
        "launch_bias": "low",
        "spin_bias": "low",
    },
    {
        "name": "Mitsubishi Tensei 1K Pro Red",
        "speed_range": (65, 82),
        "launch_bias": "high",
        "spin_bias": "mid_high",
    },
    {
        "name": "Mitsubishi Tensei 1K Pro Blue",
        "speed_range": (80, 96),
        "launch_bias": "mid",
        "spin_bias": "mid",
    },
    {
        "name": "Mitsubishi Tensei 1K Pro White",
        "speed_range": (92, 108),
        "launch_bias": "low",
        "spin_bias": "low",
    },
    {
        "name": "Mitsubishi Tensei 1K Pro Orange",
        "speed_range": (106, 125),
        "launch_bias": "low",
        "spin_bias": "low",
    },
    {
        "name": "Project X HZRDUS Smoke",
        "speed_range": (95, 115),
        "launch_bias": "low",
        "spin_bias": "low",
    },
    {
        "name": "Fujikura Speeder NX",
        "speed_range": (82, 98),
        "launch_bias": "mid_high",
        "spin_bias": "mid",
    },
]


ARCHETYPES = [
    {
        "name": "beginner_slice",
        "weight": 20,
        "gender": ["남성", "여성"],
        "age_range": (24, 52),
        "handicap_range": (22, 36),
        "speed_range": (68, 88),
        "shape_choices": ["슬라이스", "페이드"],
        "path_choices": ["아웃투인", "스트레이트"],
        "miss_choices": ["오른쪽", "양쪽 균등"],
        "concern_pool": ["거리 부족", "슬라이스", "정확도"],
    },
    {
        "name": "intermediate_balanced",
        "weight": 22,
        "gender": ["남성", "여성"],
        "age_range": (28, 48),
        "handicap_range": (12, 22),
        "speed_range": (82, 98),
        "shape_choices": ["스트레이트", "페이드", "드로우"],
        "path_choices": ["스트레이트", "아웃투인", "인투아웃"],
        "miss_choices": ["양쪽 균등", "왼쪽", "오른쪽"],
        "concern_pool": ["거리 부족", "정확도", "일관성", "탄도 조절"],
    },
    {
        "name": "senior_control",
        "weight": 12,
        "gender": ["남성", "여성"],
        "age_range": (58, 76),
        "handicap_range": (14, 28),
        "speed_range": (62, 82),
        "shape_choices": ["페이드", "슬라이스", "스트레이트"],
        "path_choices": ["아웃투인", "스트레이트"],
        "miss_choices": ["오른쪽", "양쪽 균등"],
        "concern_pool": ["거리 부족", "일관성", "정확도"],
    },
    {
        "name": "skilled_fader",
        "weight": 14,
        "gender": ["남성"],
        "age_range": (24, 42),
        "handicap_range": (0, 9),
        "speed_range": (98, 112),
        "shape_choices": ["페이드", "스트레이트"],
        "path_choices": ["아웃투인", "스트레이트"],
        "miss_choices": ["왼쪽", "양쪽 균등"],
        "concern_pool": ["정확도", "탄도 조절"],
    },
    {
        "name": "power_draw",
        "weight": 12,
        "gender": ["남성"],
        "age_range": (22, 40),
        "handicap_range": (4, 16),
        "speed_range": (104, 122),
        "shape_choices": ["드로우", "훅", "스트레이트"],
        "path_choices": ["인투아웃", "스트레이트"],
        "miss_choices": ["왼쪽", "양쪽 균등"],
        "concern_pool": ["정확도", "탄도 조절", "일관성"],
    },
    {
        "name": "women_mid_speed",
        "weight": 12,
        "gender": ["여성"],
        "age_range": (26, 56),
        "handicap_range": (10, 28),
        "speed_range": (66, 90),
        "shape_choices": ["스트레이트", "페이드", "슬라이스"],
        "path_choices": ["스트레이트", "아웃투인"],
        "miss_choices": ["오른쪽", "양쪽 균등"],
        "concern_pool": ["거리 부족", "정확도", "일관성"],
    },
    {
        "name": "low_spin_fast",
        "weight": 8,
        "gender": ["남성"],
        "age_range": (24, 44),
        "handicap_range": (0, 12),
        "speed_range": (108, 124),
        "shape_choices": ["드로우", "스트레이트", "훅"],
        "path_choices": ["인투아웃", "스트레이트"],
        "miss_choices": ["왼쪽", "양쪽 균등"],
        "concern_pool": ["정확도", "탄도 조절"],
    },
]


EXPERIENCE_BUCKETS = [
    "1년 미만",
    "1~3년",
    "3~5년",
    "5~10년",
    "10년 이상",
]

ROUND_FREQUENCY_BUCKETS = [
    "월 1~2회",
    "월 3~4회",
    "주 2회 이상",
]

TRAJECTORY_OPTIONS = ["로우", "미드", "하이"]
IMPACT_OPTIONS = ["토 쪽", "중앙", "힐 쪽"]

STATUS_MAP = {
    "low": "권장 범위보다 낮은 편",
    "ok": "권장 범위 안",
    "high": "권장 범위보다 높은 편",
}


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def choose_archetype(rng: random.Random) -> Dict[str, object]:
    weights = [item["weight"] for item in ARCHETYPES]
    return rng.choices(ARCHETYPES, weights=weights, k=1)[0]


def get_speed_window(club_speed: int) -> Dict[str, object]:
    for window in SPEED_WINDOWS:
        if window["min_speed"] <= club_speed <= window["max_speed"]:
            return window
    return SPEED_WINDOWS[-1]


def status_against_window(value: float, low: float, high: float) -> str:
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "ok"


def derive_experience(handicap: int, age: int, rng: random.Random) -> str:
    if handicap <= 6:
        return rng.choice(EXPERIENCE_BUCKETS[-2:])
    if handicap <= 14:
        return rng.choice(EXPERIENCE_BUCKETS[2:])
    if age >= 58:
        return rng.choice(EXPERIENCE_BUCKETS[1:])
    return rng.choice(EXPERIENCE_BUCKETS[:4])


def derive_round_frequency(handicap: int, rng: random.Random) -> str:
    if handicap <= 8:
        return rng.choice(ROUND_FREQUENCY_BUCKETS[1:])
    if handicap <= 18:
        return rng.choice(ROUND_FREQUENCY_BUCKETS)
    return rng.choice(ROUND_FREQUENCY_BUCKETS[:2])


def derive_glove_size(hand_size: float) -> str:
    if hand_size < 17.2:
        return "S"
    if hand_size < 18.5:
        return "M"
    if hand_size < 20.0:
        return "ML"
    if hand_size < 21.5:
        return "L"
    return "XL"


def derive_shaft_flex(club_speed: int) -> Tuple[str, str]:
    if club_speed >= 110:
        return "X", "Extra Stiff"
    if club_speed >= 97:
        return "S", "Stiff"
    if club_speed >= 84:
        return "R", "Regular"
    if club_speed >= 72:
        return "A", "Senior"
    return "L", "Ladies"


def derive_shaft_weight(club_speed: int) -> str:
    if club_speed >= 105:
        return "65g~75g"
    if club_speed >= 95:
        return "60g~70g"
    if club_speed >= 85:
        return "50g~60g"
    if club_speed >= 75:
        return "45g~55g"
    return "40g~50g"


def format_loft_window(center: float) -> str:
    center = clamp(center, 8.0, 14.0)
    return f"{center - 0.5:.1f}°~{center + 0.5:.1f}°"


def recommend_driver_loft(
    club_speed: int,
    launch_angle: float,
    attack_angle: float,
    trajectory_preference: str,
) -> str:
    if club_speed >= 110:
        center = 9.0
    elif club_speed >= 100:
        center = 9.5
    elif club_speed >= 90:
        center = 10.5
    elif club_speed >= 80:
        center = 11.5
    else:
        center = 12.5

    window = get_speed_window(club_speed)
    launch_low, launch_high = window["launch_range"]

    if launch_angle < launch_low:
        center += 0.5
    elif launch_angle > launch_high:
        center -= 0.5

    if attack_angle < 0 and club_speed < 95:
        center += 0.5

    if trajectory_preference == "하이":
        center += 0.5
    elif trajectory_preference == "로우":
        center -= 0.5

    return format_loft_window(center)


def recommend_length_and_lie(height: int, wrist_to_floor: float) -> Tuple[str, str, str]:
    baseline_wtf = height * 0.48
    diff = wrist_to_floor - baseline_wtf

    if diff > 3.0:
        length_adjustment = "+0.5\"~+1.0\""
    elif diff > 1.5:
        length_adjustment = "+0.25\"~+0.5\""
    elif diff < -3.0:
        length_adjustment = "-0.5\"~-1.0\""
    elif diff < -1.5:
        length_adjustment = "-0.25\"~-0.5\""
    else:
        length_adjustment = "표준"

    # PING 컬러코드 10단계 (DATA_RESEARCH.md §C)
    # 기준: 키 177cm + WTF 85cm = Black(표준)
    # 키 5cm당 ≈ 1단계, WTF 3cm당 ≈ 1단계
    h_score = (height - 177) / 5.0
    w_score = (wrist_to_floor - 85) / 3.0
    step = max(-4, min(5, round((h_score + w_score) / 2.0)))

    PING_COLORS = {
        -4: ("Orange", "-4° 플랫"),
        -3: ("Gold",   "-3° 플랫"),
        -2: ("Brown",  "-2° 플랫"),
        -1: ("Blue",   "-1° 플랫"),
         0: ("Black",  "표준"),
         1: ("Green",  "+1° 업라이트"),
         2: ("White",  "+2° 업라이트"),
         3: ("Silver", "+3° 업라이트"),
         4: ("Maroon", "+4° 업라이트"),
         5: ("Red",    "+5° 업라이트"),
    }
    ping_color, lie_adjustment = PING_COLORS[step]

    return length_adjustment, lie_adjustment, ping_color


def recommend_grip(glove_size: str, hand_size: float) -> str:
    if glove_size == "XL" or hand_size >= 22.0:
        return "점보(+1/16\")"
    if glove_size == "L" or hand_size >= 20.0:
        return "미드사이즈(+1/32\")"
    if glove_size == "S" or hand_size <= 17.0:
        return "언더사이즈(-1/32\")"
    return "표준"


def recommend_iron_type(handicap: int) -> str:
    if handicap <= 6:
        return "머슬백 / 플레이어즈 캐비티"
    if handicap <= 14:
        return "플레이어즈 캐비티 / 컴팩트 캐비티"
    if handicap <= 22:
        return "캐비티백 / 게임 임프루브먼트"
    return "게임 임프루브먼트 / 하이브리드 콤보"


def recommend_iron_shaft(club_speed: int, handicap: int) -> Tuple[str, str]:
    """DATA_RESEARCH.md §D 기반 구체적 모델명 사용"""
    if club_speed >= 105 and handicap <= 10:
        return "True Temper Dynamic Gold S300", "120g~130g"
    if club_speed >= 100 and handicap <= 15:
        return "True Temper Dynamic Gold AMT Tour White", "105g~125g"
    if club_speed >= 95:
        return "Nippon NS Pro 950GH", "95g~105g"
    if club_speed >= 85:
        return "Nippon NS Pro 850GH", "80g~95g"
    if club_speed >= 75:
        return "Nippon NS Pro 750GH / UST Mamiya Recoil", "65g~80g"
    return "Mitsubishi Tensei CK Blue (그래파이트)", "50g~70g"


def recommend_wedges(attack_angle: float, handicap: int) -> Tuple[str, str]:
    if handicap <= 10:
        lofts = "50° / 54° / 58°"
    elif handicap <= 20:
        lofts = "50° / 56°"
    else:
        lofts = "52° / 56°"

    if attack_angle >= 2.0:
        bounce = "미드~하이 바운스 (10°~14°)"
    elif attack_angle >= -2.0:
        bounce = "미드 바운스 (8°~12°)"
    else:
        bounce = "로우~미드 바운스 (6°~10°)"

    return lofts, bounce


def recommend_putter(path_tendency: str, shot_shape: str, height: int) -> Tuple[str, str]:
    if path_tendency == "스트레이트":
        putter_type = "페이스 밸런스 말렛"
    elif path_tendency == "인투아웃" or shot_shape == "드로우":
        putter_type = "토 행 블레이드 / 미드말렛"
    else:
        putter_type = "미드말렛 / 토 행"

    if height >= 183:
        length = "35\""
    elif height >= 170:
        length = "34\""
    else:
        length = "33\""

    return putter_type, length


def score_shaft_model(
    model: Dict[str, object],
    club_speed: int,
    spin_status: str,
    trajectory_preference: str,
    concerns: List[str],
    shot_shape: str,
) -> int:
    score = 0
    min_speed, max_speed = model["speed_range"]
    if min_speed <= club_speed <= max_speed:
        score += 5
    else:
        score -= abs(club_speed - clamp(club_speed, min_speed, max_speed))

    launch_bias = model["launch_bias"]
    spin_bias = model["spin_bias"]

    if spin_status == "high" and spin_bias == "low":
        score += 4
    if spin_status == "low" and launch_bias in {"high", "mid_high"}:
        score += 4
    if trajectory_preference == "하이" and launch_bias in {"high", "mid_high"}:
        score += 2
    if trajectory_preference == "로우" and launch_bias == "low":
        score += 2
    if "거리 부족" in concerns and launch_bias in {"high", "mid_high"}:
        score += 2
    if "정확도" in concerns and spin_bias == "low":
        score += 2
    if shot_shape in {"훅", "드로우"} and spin_bias == "low":
        score += 1
    if shot_shape in {"슬라이스", "페이드"} and launch_bias in {"mid", "mid_high", "high"}:
        score += 1

    return score


def recommend_shaft_models(
    club_speed: int,
    spin_status: str,
    trajectory_preference: str,
    concerns: List[str],
    shot_shape: str,
) -> List[str]:
    scored = sorted(
        SHAFT_CATALOG,
        key=lambda item: score_shaft_model(
            item,
            club_speed,
            spin_status,
            trajectory_preference,
            concerns,
            shot_shape,
        ),
        reverse=True,
    )
    return [item["name"] for item in scored[:2]]


def build_profile(archetype: Dict[str, object], rng: random.Random) -> Dict[str, object]:
    gender = rng.choice(archetype["gender"])
    age = rng.randint(*archetype["age_range"])
    handicap = rng.randint(*archetype["handicap_range"])

    if gender == "여성":
        height = rng.randint(156, 177)
        weight = rng.randint(48, 78)
        hand_size = round(rng.uniform(16.0, 20.2), 1)
    else:
        height = rng.randint(165, 193)
        weight = rng.randint(60, 102)
        hand_size = round(rng.uniform(17.0, 22.8), 1)

    wrist_to_floor = round(height * rng.uniform(0.455, 0.505), 1)
    arm_length = round(rng.uniform(56.0, 67.0), 1)
    glove_size = derive_glove_size(hand_size)

    return {
        "gender": gender,
        "age": age,
        "handicap": handicap,
        "experience_years": derive_experience(handicap, age, rng),
        "round_frequency": derive_round_frequency(handicap, rng),
        "height": height,
        "weight": weight,
        "wrist_to_floor": wrist_to_floor,
        "hand_size": hand_size,
        "glove_size": glove_size,
        "arm_length": arm_length,
        "dominant_hand": rng.choice(["오른손", "왼손"]) if rng.random() < 0.08 else "오른손",
    }


def build_swing(
    archetype: Dict[str, object],
    profile: Dict[str, object],
    rng: random.Random,
) -> Dict[str, object]:
    club_speed = rng.randint(*archetype["speed_range"])
    speed_window = get_speed_window(club_speed)

    launch_low, launch_high = speed_window["launch_range"]
    spin_low, spin_high = speed_window["spin_range"]
    attack_low, attack_high = speed_window["attack_range"]

    handicap = profile["handicap"]
    variation_scale = 1.4 if handicap <= 8 else 2.0 if handicap <= 18 else 3.0

    shot_shape = rng.choice(archetype["shape_choices"])
    path_tendency = rng.choice(archetype["path_choices"])
    miss_direction = rng.choice(archetype["miss_choices"])
    trajectory_preference = rng.choice(TRAJECTORY_OPTIONS)
    impact_location = rng.choice(IMPACT_OPTIONS)

    concern_count = rng.randint(1, min(3, len(archetype["concern_pool"])))
    concerns = rng.sample(archetype["concern_pool"], concern_count)

    if shot_shape == "슬라이스":
        concerns = list(dict.fromkeys(concerns + ["슬라이스"]))
    if profile["handicap"] >= 18 and "일관성" not in concerns and rng.random() < 0.5:
        concerns.append("일관성")

    smash_factor = round(rng.uniform(1.33, 1.49), 2)
    if handicap <= 8:
        smash_factor = round(rng.uniform(1.43, 1.50), 2)
    elif handicap >= 24:
        smash_factor = round(rng.uniform(1.30, 1.44), 2)

    if shot_shape == "슬라이스":
        launch_angle = rng.uniform(launch_low - 2.0, launch_high - 0.5)
        spin_rate = rng.randint(spin_high, spin_high + 1000)
    elif shot_shape == "훅":
        launch_angle = rng.uniform(launch_low - 1.0, launch_high + 0.5)
        spin_rate = rng.randint(max(1600, spin_low - 500), spin_high)
    else:
        launch_angle = rng.uniform(launch_low - variation_scale, launch_high + variation_scale)
        spin_rate = rng.randint(max(1600, spin_low - 400), spin_high + 700)

    if path_tendency == "아웃투인":
        attack_angle = rng.uniform(attack_low - 4.0, attack_high - 1.0)
    elif path_tendency == "인투아웃":
        attack_angle = rng.uniform(attack_low - 0.5, attack_high + 1.5)
    else:
        attack_angle = rng.uniform(attack_low - 2.0, attack_high + 1.0)

    if trajectory_preference == "하이":
        launch_angle += 0.8
        spin_rate += 150
    elif trajectory_preference == "로우":
        launch_angle -= 0.8
        spin_rate -= 120

    launch_angle = round(clamp(launch_angle, 8.0, 19.0), 1)
    attack_angle = round(clamp(attack_angle, -6.0, 7.0), 1)
    spin_rate = int(clamp(spin_rate, 1600, 4200))

    ball_speed = int(round(club_speed * smash_factor))
    carry_distance = int(
        round(
            club_speed * 2.12
            + (smash_factor - 1.4) * 55
            + (launch_angle - 12) * 1.8
            - abs(spin_rate - ((spin_low + spin_high) / 2)) / 220
            + rng.uniform(-6, 8)
        )
    )
    carry_distance = int(clamp(carry_distance, 145, 325))

    peak_height = int(round(clamp(launch_angle * 2.2 + rng.uniform(3, 10), 18, 42)))
    face_angle = round(rng.uniform(-4.5, 4.5), 1)
    club_path = round(
        rng.uniform(-5.0, -1.0) if path_tendency == "아웃투인"
        else rng.uniform(1.0, 5.0) if path_tendency == "인투아웃"
        else rng.uniform(-1.2, 1.2),
        1,
    )

    return {
        "club_speed": club_speed,
        "ball_speed": ball_speed,
        "smash_factor": smash_factor,
        "launch_angle": launch_angle,
        "spin_rate": spin_rate,
        "attack_angle": attack_angle,
        "dynamic_loft": round(launch_angle + rng.uniform(0.5, 3.5), 1),
        "face_angle": face_angle,
        "club_path": club_path,
        "carry_distance": carry_distance,
        "peak_height": peak_height,
        "shot_shape": shot_shape,
        "swing_path_tendency": path_tendency,
        "miss_direction": miss_direction,
        "impact_location": impact_location,
        "trajectory_preference": trajectory_preference,
        "concerns": concerns,
    }


def build_recommendations(
    profile: Dict[str, object],
    swing: Dict[str, object],
    rng: random.Random,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    window = get_speed_window(swing["club_speed"])
    launch_low, launch_high = window["launch_range"]
    spin_low, spin_high = window["spin_range"]
    attack_low, attack_high = window["attack_range"]

    launch_status = status_against_window(swing["launch_angle"], launch_low, launch_high)
    spin_status = status_against_window(swing["spin_rate"], spin_low, spin_high)
    attack_status = status_against_window(swing["attack_angle"], attack_low, attack_high)

    shaft_flex, shaft_flex_name = derive_shaft_flex(swing["club_speed"])
    shaft_weight = derive_shaft_weight(swing["club_speed"])
    driver_loft = recommend_driver_loft(
        swing["club_speed"],
        swing["launch_angle"],
        swing["attack_angle"],
        swing["trajectory_preference"],
    )
    shaft_models = recommend_shaft_models(
        swing["club_speed"],
        spin_status,
        swing["trajectory_preference"],
        swing["concerns"],
        swing["shot_shape"],
    )
    length_adjustment, lie_adjustment, ping_color = recommend_length_and_lie(
        profile["height"],
        profile["wrist_to_floor"],
    )
    grip_size = recommend_grip(profile["glove_size"], profile["hand_size"])
    iron_type = recommend_iron_type(profile["handicap"])
    iron_shaft, iron_shaft_weight = recommend_iron_shaft(
        swing["club_speed"],
        profile["handicap"],
    )
    wedge_lofts, wedge_bounce = recommend_wedges(
        swing["attack_angle"],
        profile["handicap"],
    )
    putter_type, putter_length = recommend_putter(
        swing["swing_path_tendency"],
        swing["shot_shape"],
        profile["height"],
    )

    fit_score = 63
    if launch_status == "ok":
        fit_score += 8
    elif launch_status == "low":
        fit_score += 2
    if spin_status == "ok":
        fit_score += 8
    elif spin_status == "high":
        fit_score += 1
    if attack_status == "ok":
        fit_score += 5
    if swing["smash_factor"] >= 1.45:
        fit_score += 8
    elif swing["smash_factor"] >= 1.40:
        fit_score += 4
    if profile["handicap"] <= 8:
        fit_score += 2

    fit_score = int(clamp(fit_score + rng.randint(-3, 4), 48, 96))

    research_snapshot = {
        "speed_window": window["label"],
        "optimal_launch_range": f"{launch_low:.1f}°~{launch_high:.1f}°",
        "optimal_spin_range": f"{spin_low}~{spin_high}rpm",
        "optimal_attack_range": f"{attack_low:.1f}°~{attack_high:.1f}°",
        "launch_status": launch_status,
        "spin_status": spin_status,
        "attack_status": attack_status,
        "ping_color_code": ping_color,
    }

    recommendations = {
        "fit_score": fit_score,
        "driver_loft": driver_loft,
        "shaft_flex": shaft_flex,
        "shaft_flex_name": shaft_flex_name,
        "shaft_weight": shaft_weight,
        "shaft_models": shaft_models,
        "length_adjustment": length_adjustment,
        "lie_adjustment": lie_adjustment,
        "grip_size": grip_size,
        "iron_type": iron_type,
        "iron_shaft": iron_shaft,
        "iron_shaft_weight": iron_shaft_weight,
        "wedge_lofts": wedge_lofts,
        "wedge_bounce": wedge_bounce,
        "putter_type": putter_type,
        "putter_length": putter_length,
    }

    return recommendations, research_snapshot


def build_user_prompt(
    profile: Dict[str, object],
    swing: Dict[str, object],
    recommendations: Dict[str, object],
    research_snapshot: Dict[str, object],
) -> str:
    return (
        "다음 골퍼 데이터를 바탕으로 전문 피팅 코멘트를 작성해주세요.\n\n"
        "[골퍼 프로필]\n"
        f"- 성별/나이: {profile['gender']} / {profile['age']}세\n"
        f"- 핸디캡: {profile['handicap']} | 경력: {profile['experience_years']} | 라운드 빈도: {profile['round_frequency']}\n"
        f"- 키/몸무게: {profile['height']}cm / {profile['weight']}kg\n"
        f"- 손목~바닥: {profile['wrist_to_floor']}cm | 손 크기: {profile['hand_size']}cm | 장갑: {profile['glove_size']}\n\n"
        "[스윙 데이터]\n"
        f"- 클럽 스피드: {swing['club_speed']}mph | 볼 스피드: {swing['ball_speed']}mph | 스매시 팩터: {swing['smash_factor']}\n"
        f"- 발사각: {swing['launch_angle']}° | 스핀: {swing['spin_rate']}rpm | 어택 앵글: {swing['attack_angle']}°\n"
        f"- 캐리: {swing['carry_distance']}yd | 최고점: {swing['peak_height']}yd\n"
        f"- 구질: {swing['shot_shape']} | 스윙 패스 경향: {swing['swing_path_tendency']} | 미스 방향: {swing['miss_direction']}\n"
        f"- 임팩트 위치: {swing['impact_location']} | 선호 탄도: {swing['trajectory_preference']}\n"
        f"- 고민 사항: {', '.join(swing['concerns'])}\n\n"
        "[연구 기준 요약]\n"
        f"- 권장 발사각 범위: {research_snapshot['optimal_launch_range']}\n"
        f"- 권장 스핀 범위: {research_snapshot['optimal_spin_range']}\n"
        f"- 권장 어택 앵글 범위: {research_snapshot['optimal_attack_range']}\n"
        f"- 현재 상태: 발사각 {STATUS_MAP[research_snapshot['launch_status']]}, "
        f"스핀 {STATUS_MAP[research_snapshot['spin_status']]}, "
        f"어택 앵글 {STATUS_MAP[research_snapshot['attack_status']]}\n\n"
        "[계산된 추천 스펙]\n"
        f"- 드라이버 로프트: {recommendations['driver_loft']}\n"
        f"- 드라이버 샤프트: {recommendations['shaft_flex']} ({recommendations['shaft_flex_name']}) / {recommendations['shaft_weight']}\n"
        f"- 추천 샤프트 모델: {', '.join(recommendations['shaft_models'])}\n"
        f"- 길이 조정: {recommendations['length_adjustment']} | 라이각 조정: {recommendations['lie_adjustment']}\n"
        f"- 그립: {recommendations['grip_size']}\n"
        f"- 아이언 타입: {recommendations['iron_type']}\n"
        f"- 아이언 샤프트: {recommendations['iron_shaft']} / {recommendations['iron_shaft_weight']}\n"
        f"- 웨지: {recommendations['wedge_lofts']} / {recommendations['wedge_bounce']}\n"
        f"- 퍼터: {recommendations['putter_type']} / {recommendations['putter_length']}\n\n"
        "아래 형식으로 답변해주세요:\n"
        "1. 종합 진단\n"
        "2. 추천 스펙 요약\n"
        "3. 추천 이유\n"
        "4. 시타 시 우선 확인할 포인트"
    )


def _get_surefit_tips(shot_shape: str, path_tendency: str, trajectory_preference: str) -> List[str]:
    """Titleist SureFit A1~D4 방향 가이드 (DATA_RESEARCH.md §C)"""
    tips = []
    if trajectory_preference == "하이":
        tips.append("Titleist SureFit 기준으로 로프트+ 방향(차트 위쪽)을 설정하면 탄도를 더 높일 수 있습니다.")
    elif trajectory_preference == "로우":
        tips.append("SureFit 차트 아래쪽(로프트-) 설정으로 낮은 탄도와 런 거리를 확보할 수 있습니다.")

    if shot_shape in ("슬라이스", "페이드"):
        tips.append(
            "SureFit 차트 좌측(드로우 방향) 설정이 오픈 페이스 경향을 완화합니다. "
            f"PING {path_tendency}패스 특성상 라이각 조정과 병행하면 효과적입니다."
        )
    elif shot_shape in ("훅", "드로우") and path_tendency == "인투아웃":
        tips.append("SureFit 차트 우측(페이드 방향) 설정으로 과도한 드로우를 억제할 수 있습니다.")
    return tips


def build_assistant_reply(
    profile: Dict[str, object],
    swing: Dict[str, object],
    recommendations: Dict[str, object],
    research_snapshot: Dict[str, object],
) -> str:
    launch_comment = (
        f"{swing['club_speed']}mph 구간의 권장 발사각은 {research_snapshot['optimal_launch_range']}인데 "
        f"현재 {swing['launch_angle']}°로 {STATUS_MAP[research_snapshot['launch_status']]}입니다."
    )
    spin_comment = (
        f"권장 스핀 범위는 {research_snapshot['optimal_spin_range']}이며 "
        f"현재 {swing['spin_rate']}rpm으로 {STATUS_MAP[research_snapshot['spin_status']]}입니다."
    )
    attack_comment = (
        f"어택 앵글은 {swing['attack_angle']}°로 "
        f"{STATUS_MAP[research_snapshot['attack_status']]} 쪽에 있습니다."
    )

    reason_lines = [
        f"드라이버는 {recommendations['driver_loft']}와 {recommendations['shaft_flex']} 플렉스 조합으로 시작하는 것이 무난합니다.",
        f"샤프트는 {', '.join(recommendations['shaft_models'])}처럼 현재 스피드 구간과 탄도 성향에 맞는 라인업이 적합합니다.",
        f"체형 기준으로 길이는 {recommendations['length_adjustment']}, 라이각은 {recommendations['lie_adjustment']} 방향이 자연스럽습니다.",
        f"아이언은 {recommendations['iron_type']} 성향이 현재 핸디캡과 반복 재현성에 더 잘 맞습니다.",
        f"웨지는 {recommendations['wedge_lofts']} 구성과 {recommendations['wedge_bounce']}가 현재 어택 앵글 기준으로 안정적입니다.",
    ]

    if "슬라이스" in swing["concerns"] or swing["shot_shape"] == "슬라이스":
        reason_lines.append("슬라이스 보정이 중요하므로 로프트와 출발 방향 안정화부터 먼저 확인하는 편이 좋습니다.")
    if "거리 부족" in swing["concerns"]:
        reason_lines.append("거리 고민이 크다면 샤프트 강성보다 정타율과 출발각 개선이 먼저 체감될 가능성이 큽니다.")
    if "정확도" in swing["concerns"]:
        reason_lines.append("정확도를 우선할 때는 무리하게 저스핀 세팅으로 가기보다 분산이 줄어드는지 먼저 체크해야 합니다.")

    # Titleist SureFit 방향 팁 (DATA_RESEARCH.md §C)
    surefit_tips = _get_surefit_tips(swing["shot_shape"], swing["swing_path_tendency"], swing["trajectory_preference"])
    reason_lines.extend(surefit_tips)

    next_steps = [
        f"첫 번째 시타 포인트는 드라이버 로프트 {recommendations['driver_loft']} 구간에서 출발각과 최고점이 안정되는지 확인하는 것입니다.",
        f"두 번째는 추천 샤프트 {', '.join(recommendations['shaft_models'])} 중에서 좌우 분산이 가장 줄어드는 모델을 찾는 것입니다.",
        f"세 번째는 라이각 {recommendations['lie_adjustment']} 조정 후 아이언 시작 방향과 잔디 상호작용이 좋아지는지 확인하는 것입니다.",
    ]

    return (
        "1. 종합 진단\n"
        f"- 현재 FIT SCORE는 {recommendations['fit_score']}점 수준으로, 기본 조합은 맞지만 미세 조정 여지가 남아 있습니다.\n"
        f"- {launch_comment}\n"
        f"- {spin_comment}\n"
        f"- {attack_comment}\n\n"
        "2. 추천 스펙 요약\n"
        f"- 드라이버: {recommendations['driver_loft']} / {recommendations['shaft_flex']} ({recommendations['shaft_flex_name']}) / {recommendations['shaft_weight']}\n"
        f"- 추천 샤프트 모델: {', '.join(recommendations['shaft_models'])}\n"
        f"- 길이/라이/그립: {recommendations['length_adjustment']} / {recommendations['lie_adjustment']} / {recommendations['grip_size']}\n"
        f"- 아이언: {recommendations['iron_type']} / {recommendations['iron_shaft']} ({recommendations['iron_shaft_weight']})\n"
        f"- 웨지: {recommendations['wedge_lofts']} / {recommendations['wedge_bounce']}\n"
        f"- 퍼터: {recommendations['putter_type']} / {recommendations['putter_length']}\n\n"
        "3. 추천 이유\n"
        + "\n".join(f"- {line}" for line in reason_lines)
        + "\n\n4. 시타 시 우선 확인할 포인트\n"
        + "\n".join(f"- {line}" for line in next_steps)
    )


def make_sample(sample_id: int, rng: random.Random) -> Dict[str, object]:
    archetype = choose_archetype(rng)
    profile = build_profile(archetype, rng)
    swing = build_swing(archetype, profile, rng)
    recommendations, research_snapshot = build_recommendations(profile, swing, rng)

    user_prompt = build_user_prompt(profile, swing, recommendations, research_snapshot)
    assistant_reply = build_assistant_reply(profile, swing, recommendations, research_snapshot)

    return {
        "id": f"sample_{sample_id:05d}",
        "archetype": archetype["name"],
        "profile": profile,
        "swing": swing,
        "recommendations": recommendations,
        "research_snapshot": research_snapshot,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_reply},
        ],
    }


def write_dataset(samples: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")


def preview_sample(sample: Dict[str, object]) -> None:
    print(f"[ID] {sample['id']} ({sample['archetype']})")
    print("[USER PREVIEW]")
    print(sample["messages"][1]["content"][:800])
    print("\n[ASSISTANT PREVIEW]")
    print(sample["messages"][2]["content"][:800])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FITTED SLM training data.")
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of training samples to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "golf_fitting_train.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print a sample preview after generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    samples = [make_sample(index + 1, rng) for index in range(args.samples)]
    write_dataset(samples, args.output)

    print(f"Generated {len(samples)} samples.")
    print(f"Saved to {args.output}")

    if args.preview and samples:
        print()
        preview_sample(samples[0])


if __name__ == "__main__":
    main()
