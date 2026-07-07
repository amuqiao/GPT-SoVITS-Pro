#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage / 用法:
  check_server_env.sh [-h|--help]

一句话职责:
  跨平台(Linux / macOS)体检 AI / 音视频 / 模型训练 / GPU 类项目共用的
  "基础层"前置依赖,缺失项直接给出对应平台的安装命令。只管通用地基,
  不检查任一项目自己的 pip 包、模型文件或版本锁——那些交给项目自身的
  安装方式。

Options / 选项:
  -h, --help   显示本帮助并退出。

检查范围(固定,不随项目膨胀):
  OS / 架构、CPU / 内存 / 磁盘、GPU(Linux CUDA / macOS Metal-MPS)、
  FFmpeg、Python、pip 或 uv、git、curl、docker(仅信息)、网络连通性。

不负责:
  不读取项目 pyproject.toml / requirements、不逐一验证 pip 依赖、
  不检查具体模型文件、不锁定 torch/CUDA 版本、不做 per-project 配置。

运行环境:
  依赖 bash;Linux 用 nvidia-smi/lspci/nproc/free 等,macOS 用
  sysctl/sw_vers。缺失的必要基础项在报告末尾统一列出安装命令。

输出:
  完整人读报告(含缺失项与安装命令)输出到 stdout,可安全重定向到文件;
  脚本级错误(如非法参数)输出到 stderr。

副作用与保护边界:
  只读体检,不安装、不修改系统、不写文件;仅对 pypi 和 huggingface 各做
  一次只读 HEAD 连通性探测,不下载内容、不打印任何密钥或凭据。

常用示例:
  check_server_env.sh
  check_server_env.sh > server-env.txt

Exit Codes:
  0  全部必要基础依赖就绪。
  1  存在缺失的必要基础依赖(报告末尾已给出安装命令)。
  2  参数或用法错误。
EOF
}

section() {
  printf '\n===== %s =====\n' "$1"
}

print_kv() {
  printf '%-24s %s\n' "$1:" "$2"
}

bytes_to_gib() {
  awk -v bytes="$1" 'BEGIN { printf "%.1f GiB", bytes / 1024 / 1024 / 1024 }'
}

# 展示型探测:命令缺失标记 MISSING;命令存在但返回非 0 时吞状态,
# 避免 errexit 中断整份报告(收集失败信息本身就是本脚本的意图)。
run_cmd() {
  local label="$1"
  shift
  printf '\n# %s\n$ %s\n' "$label" "$*"
  if command -v "$1" >/dev/null 2>&1; then
    "$@" 2>&1 || true
  else
    printf 'MISSING: command not found: %s\n' "$1"
  fi
}

run_cmd_head() {
  local label="$1"
  local lines="$2"
  shift 2
  printf '\n# %s\n$ %s | head -%s\n' "$label" "$*" "$lines"
  if command -v "$1" >/dev/null 2>&1; then
    "$@" 2>&1 | head -n "${lines}" || true
  else
    printf 'MISSING: command not found: %s\n' "$1"
  fi
}

# ---- 平台与包管理器识别 ----
detect_os() {
  case "$(uname -s 2>/dev/null || echo unknown)" in
    Linux) echo linux ;;
    Darwin) echo macos ;;
    *) echo other ;;
  esac
}

# 依赖全局 OS_KIND;返回用于拼装安装命令的包管理器标识。
detect_pkg_mgr() {
  case "$OS_KIND" in
    macos)
      if command -v brew >/dev/null 2>&1; then echo brew; else echo none-macos; fi
      ;;
    linux)
      local m
      for m in apt-get dnf yum pacman zypper apk; do
        if command -v "$m" >/dev/null 2>&1; then echo "$m"; return; fi
      done
      echo none-linux
      ;;
    *) echo unknown ;;
  esac
}

# 按当前平台/包管理器,给出某个基础依赖的安装命令字符串。
install_hint() {
  local pkg="$1"
  if [ "$pkg" = "uv" ]; then
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh    # 或: pip install uv / brew install uv"
    return
  fi
  case "$PKG_MGR" in
    brew)
      case "$pkg" in
        python3) echo "brew install python" ;;
        *)       echo "brew install $pkg" ;;
      esac
      ;;
    apt-get)
      case "$pkg" in
        python3) echo "sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv" ;;
        *)       echo "sudo apt-get update && sudo apt-get install -y $pkg" ;;
      esac
      ;;
    dnf|yum)
      case "$pkg" in
        python3) echo "sudo $PKG_MGR install -y python3 python3-pip" ;;
        *)       echo "sudo $PKG_MGR install -y $pkg" ;;
      esac
      ;;
    pacman)
      case "$pkg" in
        python3) echo "sudo pacman -S --noconfirm python python-pip" ;;
        *)       echo "sudo pacman -S --noconfirm $pkg" ;;
      esac
      ;;
    zypper)
      case "$pkg" in
        python3) echo "sudo zypper install -y python3 python3-pip" ;;
        *)       echo "sudo zypper install -y $pkg" ;;
      esac
      ;;
    apk)
      case "$pkg" in
        python3) echo "sudo apk add python3 py3-pip" ;;
        *)       echo "sudo apk add $pkg" ;;
      esac
      ;;
    none-macos)
      echo "先装 Homebrew(https://brew.sh),再执行: brew install ${pkg/python3/python}"
      ;;
    *)
      echo "未识别包管理器,请用系统包管理器手动安装: $pkg"
      ;;
  esac
}

# ---- 缺失项收集(两个并行索引数组,兼容 macOS 自带 bash 3.2) ----
MISSING_LABELS=()
MISSING_CMDS=()
note_missing() {
  MISSING_LABELS+=("$1")
  MISSING_CMDS+=("$2")
}

# 检查一个必要基础命令:存在则打印首行版本,缺失则记入待安装清单。
check_required() {
  local label="$1" cmd="$2" pkg="$3" ver_flag="${4:---version}"
  printf '\n# %s\n$ %s %s\n' "$label" "$cmd" "$ver_flag"
  if command -v "$cmd" >/dev/null 2>&1; then
    "$cmd" "$ver_flag" 2>&1 | head -n 1 || true
  else
    printf 'MISSING: 未安装 %s\n' "$cmd"
    note_missing "$label ($cmd)" "$(install_hint "$pkg")"
  fi
}

check_url() {
  local label="$1" url="$2"
  printf '\n# %s\n$ curl -sS -I --max-time 8 %s\n' "$label" "$url"
  if command -v curl >/dev/null 2>&1; then
    curl -sS -I --max-time 8 "$url" 2>&1 | head -n 5 || true
  else
    printf 'SKIP: curl 缺失,跳过 %s 连通性探测\n' "$url"
  fi
}

# ---- -h|--help 早拦截;非法参数 fail-fast 返回 2 ----
while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      printf 'ERROR: 未知参数: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
    *)
      printf 'ERROR: 本脚本不接受位置参数: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

OS_KIND="$(detect_os)"
PKG_MGR="$(detect_pkg_mgr)"

section "SUMMARY"
print_kv "生成时间 Generated at" "$(date '+%Y-%m-%d %H:%M:%S %Z' 2>/dev/null || date)"
print_kv "主机名 Hostname" "$(hostname 2>/dev/null || printf 'UNKNOWN')"
print_kv "当前用户 User" "$(id -un 2>/dev/null || whoami 2>/dev/null || printf 'UNKNOWN')"
print_kv "平台 Platform" "${OS_KIND}"
print_kv "架构 Arch" "$(uname -m 2>/dev/null || printf 'UNKNOWN')"
print_kv "包管理器 Pkg manager" "${PKG_MGR}"

section "系统 / 内核 OS / KERNEL"
if [ "$OS_KIND" = "linux" ]; then
  if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    print_kv "发行版 Distribution" "${PRETTY_NAME:-UNKNOWN}"
    print_kv "ID" "${ID:-UNKNOWN}"
    print_kv "版本 Version ID" "${VERSION_ID:-UNKNOWN}"
  else
    printf 'MISSING: /etc/os-release 不存在或不可读\n'
  fi
elif [ "$OS_KIND" = "macos" ]; then
  print_kv "系统 macOS" "$(sw_vers -productName 2>/dev/null || echo macOS) $(sw_vers -productVersion 2>/dev/null || echo UNKNOWN)"
  print_kv "构建 Build" "$(sw_vers -buildVersion 2>/dev/null || echo UNKNOWN)"
fi
run_cmd "内核信息 kernel" uname -a

section "CPU / 内存 / 磁盘 CPU / MEMORY / DISK"
if [ "$OS_KIND" = "linux" ]; then
  print_kv "CPU 逻辑核心 cores" "$(command -v nproc >/dev/null 2>&1 && nproc || echo UNKNOWN)"
  if [ -r /proc/meminfo ]; then
    mem_kib="$(awk '/MemTotal:/ { print $2 }' /proc/meminfo || true)"
    [ -n "${mem_kib:-}" ] && print_kv "内存总量 Memory total" "$(bytes_to_gib "$((mem_kib * 1024))")" || true
  fi
  run_cmd "内存 memory" free -h
elif [ "$OS_KIND" = "macos" ]; then
  print_kv "CPU 型号 model" "$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo UNKNOWN)"
  print_kv "CPU 逻辑核心 cores" "$(sysctl -n hw.logicalcpu 2>/dev/null || echo UNKNOWN)"
  mem_bytes="$(sysctl -n hw.memsize 2>/dev/null || true)"
  [ -n "${mem_bytes:-}" ] && print_kv "内存总量 Memory total" "$(bytes_to_gib "$mem_bytes")" || true
fi
run_cmd "根分区磁盘 root disk" df -h /
if [ -d "$PWD" ]; then
  run_cmd "当前目录磁盘 current dir disk" df -h "$PWD"
fi

section "GPU / 加速 GPU / ACCELERATION"
if [ "$OS_KIND" = "macos" ]; then
  if [ "$(uname -m 2>/dev/null || echo)" = "arm64" ]; then
    print_kv "GPU" "Apple Silicon 集成 GPU($(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo Apple))"
    print_kv "加速 Acceleration" "Metal / MPS(由各项目自带的 PyTorch 提供,本脚本不检查项目内 torch)"
  else
    run_cmd_head "显示适配器 displays" 40 system_profiler SPDisplaysDataType
  fi
elif command -v nvidia-smi >/dev/null 2>&1; then
  run_cmd "NVIDIA 完整信息 nvidia full" nvidia-smi
  run_cmd "NVIDIA 简洁信息 nvidia compact" nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  driver_cuda="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([^ |]*\).*/\1/p' | head -1 || true)"
  if [ -n "${driver_cuda}" ]; then
    print_kv "驱动支持 Driver CUDA max" "${driver_cuda}"
  else
    print_kv "驱动支持 Driver CUDA max" "UNKNOWN: 无法从 nvidia-smi 输出解析"
  fi
  run_cmd "CUDA 编译器 nvcc" nvcc --version
else
  printf '\n$ nvidia-smi\n'
  printf 'MISSING: command not found: nvidia-smi\n'
  printf '含义 Meaning: 没装 NVIDIA 驱动、PATH 不完整,或本机没有 NVIDIA GPU。\n'
  if command -v lspci >/dev/null 2>&1; then
    printf '\n$ lspci | grep -i -E "vga|3d|nvidia"\n'
    lspci | grep -i -E "vga|3d|nvidia" 2>&1 || printf 'NO MATCH: lspci 未见 VGA/3D/NVIDIA 设备行\n'
  fi
fi

section "必要基础工具 REQUIRED BASE TOOLS"
check_required "FFmpeg 音视频处理" ffmpeg ffmpeg -version
check_required "Python 解释器" python3 python3 --version
check_required "Git 版本控制" git git --version
check_required "curl 下载工具" curl curl --version

# Python 包管理器:pip 或 uv 至少要有一个。
printf '\n# Python 包管理器 pip / uv\n'
have_pkg=0
if command -v pip3 >/dev/null 2>&1; then
  printf '$ pip3 --version\n'; pip3 --version 2>&1 | head -n 1 || true; have_pkg=1
elif command -v pip >/dev/null 2>&1; then
  printf '$ pip --version\n'; pip --version 2>&1 | head -n 1 || true; have_pkg=1
else
  printf 'MISSING: 未安装 pip / pip3\n'
fi
if command -v uv >/dev/null 2>&1; then
  printf '$ uv --version\n'; uv --version 2>&1 | head -n 1 || true; have_pkg=1
else
  printf 'MISSING: 未安装 uv\n'
fi
if [ "$have_pkg" -eq 0 ]; then
  note_missing "Python 包管理器 (pip / uv)" "$(install_hint uv)"
fi

section "容器工具 CONTAINER (仅信息)"
run_cmd "docker" docker --version
if command -v docker >/dev/null 2>&1; then
  run_cmd "docker compose" docker compose version
fi

section "网络连通性 NETWORK (仅信息)"
check_url "PyPI 连通性" https://pypi.org/simple/
check_url "HuggingFace 连通性" https://huggingface.co

section "体检结论 RESULT"
if [ "${#MISSING_LABELS[@]}" -eq 0 ]; then
  printf '全部必要基础依赖已就绪。All required base dependencies are present.\n'
  printf '提示: 项目自身的 pip 依赖 / 模型 / 版本锁请按各项目安装方式处理。\n'
  exit 0
else
  printf '检测到 %d 项必要基础依赖缺失,请按对应平台命令安装:\n' "${#MISSING_LABELS[@]}"
  idx=0
  while [ "$idx" -lt "${#MISSING_LABELS[@]}" ]; do
    printf '\n- 缺失 Missing: %s\n  安装 Install: %s\n' "${MISSING_LABELS[$idx]}" "${MISSING_CMDS[$idx]}"
    idx=$((idx + 1))
  done
  printf '\n补齐上述基础依赖后,再按各项目自身的安装方式处理其余依赖。\n'
  exit 1
fi
