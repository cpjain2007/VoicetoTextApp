# Deploy VoiceToText API as Firebase Functions + Firestore rules.
# Prerequisites (one-time): npm install (repo root), npm run firebase:login, create Firestore DB in console.
# Usage:
#   .\scripts\firebase-deploy.ps1 -ProjectId your-firebase-project-id
#   or:  $env:FIREBASE_PROJECT = "your-id"; .\scripts\firebase-deploy.ps1

param(
  [string] $ProjectId = $env:FIREBASE_PROJECT
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path (Join-Path $root "firebase.json"))) {
  throw "firebase.json not found under $root"
}

if (-not $ProjectId) {
  Write-Host "Set FIREBASE_PROJECT or pass -ProjectId (lowercase Firebase / GCP project id)." -ForegroundColor Yellow
  exit 1
}

Set-Location $root
Write-Host "Using project: $ProjectId" -ForegroundColor Cyan
$env:FUNCTIONS_DISCOVERY_TIMEOUT = "60"
npx firebase use $ProjectId
node (Join-Path $root "scripts\sync-functions-env.js") $ProjectId
npx firebase deploy --only "functions,firestore:rules"
Write-Host "Done. Set function env vars in Console (Functions > api > Environment variables) or gcloud." -ForegroundColor Green
Write-Host "For public HTTP from Expo: Cloud Run > service for api > Security > Allow unauthenticated." -ForegroundColor Green
