param(
  [switch]$UseOllama,
  [string]$Model = "qwen2.5:3b-instruct",
  [int]$Port = 4173
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$env:PORT = "$Port"

if ($UseOllama) {
  $env:OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
  $env:OLLAMA_MODEL = $Model
  Write-Host "Starting FITTED local demo with Ollama model: $Model"
  Write-Host "Make sure 'ollama serve' is running first."
} else {
  if (Test-Path Env:OLLAMA_MODEL) {
    Remove-Item Env:OLLAMA_MODEL
  }
  Write-Host "Starting FITTED local demo in fallback-first mode."
  Write-Host "If Ollama is running, the server will still auto-try common local models."
}

Write-Host "Open http://127.0.0.1:$Port after the server starts."
Write-Host "Stop with Ctrl+C."

node server.js
