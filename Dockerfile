# mrs-lab — the robot policy evaluation agent, as one container.
#
# OpenClaw supplies the agent loop and the skill loader; everything the four
# pipeline skills shell out to has to be in here with it: Blender (driven over
# blender-mcp), MuJoCo, the pi0.5 inference stack, and Quarto for the reports.
#
# Blender is the reason for the X server. The BlenderMCP addon refuses to run
# under `blender -b` (addon.py: "cannot start server in background mode"), so
# Blender runs a real GUI against Xvfb with Mesa's llvmpipe behind it. That also
# gives get_viewport_screenshot the GL context its offscreen render path needs.
#
# Sizes matter here: the Maritime rootfs is provisioned from the image, and
# /data is 10 GB, which the pi0.5 checkpoint (7.5 GB) has to fit inside.

FROM ghcr.io/openclaw/openclaw:2026.5.28

USER root
ENV DEBIAN_FRONTEND=noninteractive

# --- virtual display, software GL, python ------------------------------------
#
# libgl1-mesa-dri carries llvmpipe (Blender's viewport); libosmesa6 is the
# offscreen path MuJoCo renders the policy cameras through.
RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb x11-xserver-utils x11-utils \
        libgl1 libglx-mesa0 libgl1-mesa-dri libegl1 libosmesa6 libglu1-mesa \
        libx11-6 libxext6 libxi6 libxxf86vm1 libxfixes3 libxrender1 \
        libxkbcommon0 libsm6 libice6 \
        python3 python3-venv python3-dev \
        git curl ca-certificates xz-utils procps fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# --- Blender + the BlenderMCP addon ------------------------------------------
ARG BLENDER_SERIES=4.2
ARG BLENDER_VERSION=4.2.9
RUN curl -fsSL "https://download.blender.org/release/Blender${BLENDER_SERIES}/blender-${BLENDER_VERSION}-linux-x64.tar.xz" \
        | tar -xJ -C /opt \
    && mv "/opt/blender-${BLENDER_VERSION}-linux-x64" /opt/blender \
    && ln -s /opt/blender/blender /usr/local/bin/blender

# Pinned to the revision vendored in blender-mcp/ so the container and the
# workstation run byte-identical addon code.
#
# It is staged, not installed: Blender 4.2 has no bundled scripts/addons/
# directory and does not pick the file up from BLENDER_USER_SCRIPTS either, so
# boot_blender.py hands it to Blender's own installer at runtime.
ARG BLENDER_MCP_REF=da4e16d
RUN mkdir -p /opt/blender-mcp \
    && curl -fsSL -o /opt/blender-mcp/addon.py \
        "https://raw.githubusercontent.com/ahujasid/blender-mcp/${BLENDER_MCP_REF}/addon.py"

# --- Quarto ------------------------------------------------------------------
ARG QUARTO_VERSION=1.10.18
RUN curl -fsSL -o /tmp/quarto.deb \
        "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-amd64.deb" \
    && dpkg -i /tmp/quarto.deb \
    && rm /tmp/quarto.deb

# --- python: the simulator, the policy, the report engine --------------------
#
# CPU torch on purpose — the Maritime microVM has no GPU. pi0.5 runs at roughly
# 20-30 s per action chunk on 4 vCPU against ~1 s on an accelerator; that is the
# accepted cost of keeping the whole pipeline in one place.
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir \
        "mujoco>=3.2" "numpy>=1.26" "transformers>=4.50" "safetensors>=0.4" \
        "huggingface-hub>=0.24" "imageio>=2.34" imageio-ffmpeg \
        matplotlib pandas jupyter blender-mcp \
    && python3 -m ipykernel install --sys-prefix --name python3 --display-name python3

# --- the project itself ------------------------------------------------------
COPY . /opt/mrs
RUN pip install --no-cache-dir -e /opt/mrs \
    && chmod +x /opt/mrs/deploy/entrypoint.sh /opt/mrs/deploy/bin/*

ENV PATH=/opt/mrs/deploy/bin:$PATH \
    DISPLAY=:99 \
    LIBGL_ALWAYS_SOFTWARE=1 \
    GALLIUM_DRIVER=llvmpipe \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    BLENDER_HOST=127.0.0.1 \
    BLENDER_PORT=9876 \
    OPENCLAW_HEADLESS=true \
    HF_HOME=/data/hf \
    MRS_LAB=/data/lab \
    MRS_ASSET_DIR=/data/lab/.cache \
    MPLBACKEND=Agg \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app
ENTRYPOINT ["tini", "-s", "--", "/opt/mrs/deploy/entrypoint.sh"]
CMD ["node", "openclaw.mjs", "gateway"]
