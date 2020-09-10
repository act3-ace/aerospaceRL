ARG BASE_IMAGE=nvidia/cuda
ARG BASE_TAG=10.2-cudnn8-devel-ubuntu18.04


################################################################################
# Add OpenGl
################################################################################
FROM ${BASE_IMAGE}:${BASE_TAG} as GlBase

RUN \
dpkg --add-architecture i386 && \
apt-get update && apt-get install -y --no-install-recommends \
  libxau6 libxau6:i386 \
  libxdmcp6 libxdmcp6:i386 \
  libxcb1 libxcb1:i386 \
  libxext6 libxext6:i386 \
  libx11-6 libx11-6:i386 \
  libglvnd0 libglvnd0:i386 \
  libgl1 libgl1:i386 \
  libglx0 libglx0:i386 \
  libegl1 libegl1:i386 \
  libgles2 libgles2:i386 \
  pkg-config \
  libglvnd-dev libglvnd-dev:i386 \
  libgl1-mesa-dev libgl1-mesa-dev:i386 \
  libegl1-mesa-dev libegl1-mesa-dev:i386 \
  libgles2-mesa-dev libgles2-mesa-dev:i386 && \
rm -rf /var/lib/apt/lists/*

COPY ./docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
 ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compat32,utility

RUN \
echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

# Required for non-glvnd setups.
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

################################################################################
# Add User
# Create a user for the image can be overwritten with docker args
################################################################################
FROM GlBase as UserBase

ARG NEW_UID=1000
ARG NEW_GID=1000
ARG NEW_GROUP=algo
ARG NEW_USER=coder

# Install basic utilities
RUN \
apt-get update -y && \
apt-get install -y --no-install-recommends sudo && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

RUN \
addgroup --gid $NEW_GID $NEW_GROUP && \
adduser --uid $NEW_UID --gid $NEW_GID --disabled-password --gecos '' $NEW_USER && \
usermod -aG sudo ${NEW_USER} && \
echo "${NEW_USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

###############################################################################
# Install fixuid
###############################################################################
FROM UserBase as FixUid

ARG NEW_GROUP=algo
ARG NEW_USER=coder

RUN \
apt-get update -y && \
apt-get install -y --no-install-recommends curl && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

RUN \
ARCH="$(dpkg --print-architecture)" && \
curl -fsSL "https://github.com/boxboat/fixuid/releases/download/v0.4.1/fixuid-0.4.1-linux-$ARCH.tar.gz" | tar -C /usr/local/bin -xzf - && \
chown root:root /usr/local/bin/fixuid && \
chmod 4755 /usr/local/bin/fixuid && \
mkdir -p /etc/fixuid && \
printf "user: $NEW_USER\ngroup: $NEW_GROUP\n" > /etc/fixuid/config.yml

###############################################################################
# Install miniconda
###############################################################################
FROM UserBase as Miniconda

ARG NEW_GROUP=algo
ARG NEW_USER=coder

ARG PYTHON_VERSION=3.7
ARG CONDA_PYTHON_VERSION=3
ARG CONDA_DIR=/opt/conda

# Install basic utilities
RUN \
apt-get update -y && \
apt-get install -y --no-install-recommends \
  git \
  wget \
  unzip \
  bzip2 \
  build-essential \
  ca-certificates \
  apt-utils && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

WORKDIR /opt/temp

# Install miniconda
ENV PATH $CONDA_DIR/bin:$PATH

RUN \
wget \
  --quiet \
  https://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh \
  -O /opt/temp/miniconda.sh && \
echo "export PATH=$CONDA_DIR/bin:$PATH" > /etc/profile.d/conda.sh && \
/bin/bash /opt/temp/miniconda.sh -b -p $CONDA_DIR  && \
rm /opt/temp/miniconda.sh && \
${CONDA_DIR}/bin/conda clean -tipsy

RUN chown -R ${NEW_USER} ${CONDA_DIR}

###############################################################################
# Install Source
###############################################################################
FROM FixUid as AeroEnv

ARG NEW_GROUP=algo
ARG NEW_USER=coder
ARG CONDA_DIR=/opt/conda

# Miniconda
COPY --chown=${NEW_USER}:${NEW_GROUP} --from=MiniConda ${CONDA_DIR}/. ${CONDA_DIR}/
ENV PATH ${CONDA_DIR}/bin:$PATH
RUN conda init bash

USER ${NEW_USER}

RUN \
sudo apt-get update && \
sudo apt-get install -y \
  libglu1-mesa-dev \
  ssh \
  libgl1-mesa-dev \
  libosmesa6-dev \
  ffmpeg \
  libopenmpi-dev \
  xvfb
COPY --chown=${NEW_USER}:${NEW_GROUP} ./ /home/coder/aerospacerl
WORKDIR /home/${NEW_USER}/aerospacerl

RUN conda install python=3.7
RUN conda config --add channels http://conda.anaconda.org/gurobi
RUN conda install gurobi
RUN pip install -e .

###############################################################################
# install VsCode
###############################################################################
FROM AeroEnv as VsCodeBase

ARG NEW_GROUP=algo
ARG NEW_USER=coder
USER root
RUN \
apt-get update && \
apt-get install -y \
  curl \
  dumb-init \
  htop \
  locales \
  man \
  nano \
  git \
  procps \
  sudo \
  vim \
  libcap2-bin \
  lsb-release && \
rm -rf /var/lib/apt/lists/*

# https://wiki.debian.org/Locale#Manually
RUN \
sed -i "s/# en_US.UTF-8/en_US.UTF-8/" /etc/locale.gen && \
locale-gen

ENV LANG=en_US.UTF-8

RUN chsh -s /bin/bash
ENV SHELL=/bin/bash

RUN \
curl -SsL https://github.com/cdr/code-server/releases/download/3.2.0/code-server-3.2.0-linux-x86_64.tar.gz | tar -C /opt -xzf - && \
mv /opt/code-server* /opt/code-server && \
ln -s /opt/code-server/code-server /usr/local/bin/code-server

USER $NEW_USER

RUN \
sudo setcap cap_net_bind_service=+ep /opt/code-server/node

EXPOSE 8080
EXPOSE 6006
WORKDIR /home/$NEW_USER
ENTRYPOINT ["dumb-init", "fixuid", "-q", "/usr/local/bin/code-server", "--cert", "--host", "0.0.0.0", "--port", "8080", "--auth", "none", "."]

###############################################################################
# VsCode Miniconda Image
###############################################################################
FROM VsCodeBase as VsCodeDevel

ARG NEW_GROUP=algo
ARG NEW_USER=coder
ARG CONDA_DIR=/opt/conda

USER $NEW_USER

RUN \
code-server --install-extension streetsidesoftware.code-spell-checker && \
code-server --install-extension bierner.markdown-mermaid && \
code-server --install-extension DavidAnson.vscode-markdownlint && \
code-server --install-extension eamodio.gitlens && \
code-server --install-extension ms-python.anaconda-extension-pack && \
code-server --install-extension ms-python.python && \
code-server --install-extension shd101wyy.markdown-preview-enhanced && \
code-server --install-extension yzhang.markdown-all-in-one

WORKDIR /home/$NEW_USER

###############################################################################
# VsCode Miniconda Image
###############################################################################
FROM VsCodeDevel as AceHub

USER root

RUN set -eux; \
  apt-get update; \
  apt-get install -y gosu; \
  rm -rf /var/lib/apt/lists/*; \
  gosu nobody true

ADD /docker/ace-hub-entrypoint.sh /ace-hub-entrypoint.sh
ENTRYPOINT ["dumb-init", "/ace-hub-entrypoint.sh", "fixuid", "-q"]
CMD ["/usr/local/bin/code-server", "--host", "0.0.0.0", "--port", "8888", "--auth", "none", "."]



