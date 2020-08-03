FROM gitpod/workspace-full:latest

USER gitpod

### Install Python 3.7.8
RUN set -x; \
    pyenv install 3.7.7
RUN set -x; \
    pyenv global 3.7.7
