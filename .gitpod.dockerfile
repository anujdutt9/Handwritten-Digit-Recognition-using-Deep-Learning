FROM gitpod/workspace-full-vnc

# Install Python 3.7.7
RUN pyenv install 3.7.7
RUN pyenv global 3.7.7

# Git LFS
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash \
    && sudo apt-get update \
    && sudo apt-get install -y \ 
        git-lfs \
    && sudo rm -rf /var/lib/apt/lists/* \
    && git lfs install
    