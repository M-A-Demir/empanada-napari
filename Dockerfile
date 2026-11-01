ARG UBUNTU_VERSION=24.04
FROM ubuntu:${UBUNTU_VERSION}

ARG UBUNTU_VERSION
ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Europe/London
ARG LOCAL_BUILD=false
ENV TZ=${TZ}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates tzdata locales \
    libgl1 libglu1-mesa libglx-mesa0 libegl1 libopengl0 mesa-utils \
    libglib2.0-0t64 libfontconfig1 libxrender1 libdbus-1-3 libxi6 \
    libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 \
    libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 \
    libxcb-xfixes0 libxcb-xinerama0 libxcb-xinput0 libxcb-xkb1 x11-utils \
    python3 python3-venv python3-dev \
    && sed -i 's/^# *\(en_GB.UTF-8\)/\1/' /etc/locale.gen && locale-gen \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip

ENV PATH="/opt/venv/bin:${PATH}" \
    QT_X11_NO_MITSHM=1 \
    QT_XCB_NO_MITSHM=1 \
    NAPARI_ASYNC=1

# container-dist/ must exist in the build context and is used when LOCAL_BUILD=true,
# otherwise empanada-napari is installed from PyPI. Note in this case 
# the package is installed separately *before* napari, to ensure the latest
# version resolves. Additional test requirements are added in 
# second install
RUN --mount=type=bind,source=container-dist,target=/tmp/dist \
    if [ "${LOCAL_BUILD}" = "true" ]; then \
        pip install --no-cache-dir /tmp/dist/*.whl; \
	else \
		pip install --no-cache-dir empanada-napari; \
    fi \
    && pip install --no-cache-dir "napari[all]" "empanada-napari[test]" \
    && pip freeze > /opt/venv/freeze.txt

COPY tests/ /tests/
RUN chmod +x /tests/run_container_tests.sh

ARG IMAGE_VERSION="dev"
ARG IMAGE_REVISION=""
ARG IMAGE_CREATED=""

LABEL org.opencontainers.image.title="Empanada-napari" \
      org.opencontainers.image.description="Ubuntu container for napari with empanada plugin" \
      org.opencontainers.image.authors="Allan Almeida, Mariam Demir, Kedar Narayan" \
      org.opencontainers.image.source="https://github.com/volume-em/empanada-napari" \
      org.opencontainers.image.documentation="https://empanada.readthedocs.io/en/latest/index.html" \
      org.opencontainers.image.licenses="BSD-3-Clause" \
      org.opencontainers.image.base.name="docker.io/library/ubuntu:${UBUNTU_VERSION}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.revision="${IMAGE_REVISION}" \
      org.opencontainers.image.created="${IMAGE_CREATED}"

# Runs the container tests (set SKIP_TESTS=true to skip), then napari
CMD ["/bin/bash", "-c", "/tests/run_container_tests.sh && exec python -m napari"]
