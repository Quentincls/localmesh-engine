$venvActivation = Join-Path $PSScriptRoot '.venv\Scripts\Activate.ps1'
if (-not (Test-Path -LiteralPath $venvActivation -PathType Leaf)) {
    throw 'Create the Python 3.12 virtual environment in .venv before running this script.'
}

& $venvActivation
$env:LOCALMESH_ROOT = Join-Path $PSScriptRoot 'LocalMesh'

# Keep GPU caches in the writable runtime and respect existing overrides.
$localMeshCache = Join-Path $env:LOCALMESH_ROOT 'cache'
if (-not $env:FLEX_GEMM_AUTOTUNE_CACHE_PATH) {
    $flexGemmCache = Join-Path $localMeshCache 'flex_gemm'
    [System.IO.Directory]::CreateDirectory($flexGemmCache) | Out-Null
    $env:FLEX_GEMM_AUTOTUNE_CACHE_PATH = Join-Path $flexGemmCache 'autotune_cache.json'
}
if (-not $env:TRITON_CACHE_DIR -and -not $env:TRITON_HOME) {
    $env:TRITON_CACHE_DIR = Join-Path $localMeshCache 'triton'
}
if (-not $env:CUDA_CACHE_PATH) {
    $env:CUDA_CACHE_PATH = Join-Path $localMeshCache 'cuda'
}
