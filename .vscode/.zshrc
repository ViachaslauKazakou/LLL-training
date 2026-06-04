#!/bin/zsh
# VS Code terminal init script for LLM-learn workspace
# This replaces .zshrc when ZDOTDIR is set

# Activate virtualenv first
source /Users/Viachaslau_Kazakou/Work/LLM-learn/.venv/bin/activate

# Load user's original zsh config from home
if [ -f ~/.zshrc ]; then
    source ~/.zshrc
fi

# Force virtualenv prefix to show in prompt (oh-my-zsh compatible)
if [ -n "$VIRTUAL_ENV" ]; then
    VENV_NAME=$(basename "$VIRTUAL_ENV")
    # Prepend to whatever prompt oh-my-zsh set
    PROMPT="%{$fg_bold[cyan]%}($VENV_NAME)%{$reset_color%} $PROMPT"
fi
