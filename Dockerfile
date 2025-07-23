# Use the NVIDIA JAX image as the base
# FROM nvcr.io/nvidia/jax:25.04-py3
FROM ghcr.io/nvidia/jax:jax-2025-06-18

# Install zsh and other required packages
RUN apt-get update && apt-get install -y \
    zsh \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Oh My Zsh and plugins
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" && \
    git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting && \
    sed -i 's/ZSH_THEME="robbyrussell"/ZSH_THEME="avit"/' ~/.zshrc

# Set up the working directory
WORKDIR /app

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH=/root/.local/bin:$PATH

# Copy dependency files
COPY pyproject.toml ./
COPY uv.lock* ./

# Install dependencies using uv with the system Python
ENV PYTHONPATH=/usr/lib/python3/dist-packages
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages -e ".[dev]"

# Set the locale for UTF-8 used for Sphinx docs
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Set the user and group ID (these will be filled in when building)
ARG USER_ID
ARG GROUP_ID

# Ensure the app directory is owned by the user
RUN chown -R ${USER_ID}:${GROUP_ID} /app

# Set Git configuration
RUN git config --global user.name "mrazomej" && \
    git config --global user.email "manuel.razo.m@gmail.com"

USER ${USER_ID}:${GROUP_ID}

# Set the default command
SHELL ["/bin/zsh", "-c"]
ENTRYPOINT ["zsh"]