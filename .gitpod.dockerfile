FROM gitpod/workspace-full:latest

USER gitpod

# Install Python 3.7.7
RUN pyenv install 3.7.7

# Git LFS
RUN build_deps="curl" \
    && sudo apt-get update \
    && sudo apt-get install -y --no-install-recommends \
        ${build_deps} ca-certificates \
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && sudo apt-get install -y --no-install-recommends \ 
        git-lfs \
    && git lfs install \
    && sudo apt-get purge -y --auto-remove \
    &&  ${build_deps} \
    && sudo rm -rf /var/lib/apt/lists/*
