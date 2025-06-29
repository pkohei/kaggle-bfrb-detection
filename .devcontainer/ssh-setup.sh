#!/bin/bash
set -e

echo "🔑 Setting up SSH configuration..."

# Check if SSH_AUTH_SOCK is available
if [ -n "$SSH_AUTH_SOCK" ]; then
    echo "✅ SSH agent socket found: $SSH_AUTH_SOCK"

    # Test SSH agent connection
    if ssh-add -l >/dev/null 2>&1; then
        echo "✅ SSH agent is working and has keys loaded"
        ssh-add -l
    else
        echo "⚠️  SSH agent is available but no keys are loaded"
        echo "   Please add your SSH keys to the agent on the host machine:"
        echo "   ssh-add ~/.ssh/your_private_key"
    fi
else
    echo "⚠️  SSH_AUTH_SOCK not found"
    echo "   SSH agent forwarding may not be configured properly"
    echo "   Make sure you have:"
    echo "   1. SSH agent running on your host"
    echo "   2. Keys added to the agent (ssh-add)"
    echo "   3. VS Code configured with proper SSH settings"
fi

# Ensure SSH directory exists and has proper permissions
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Create or update SSH config for GitHub and common Git hosts
SSH_CONFIG=~/.ssh/config
if [ ! -f "$SSH_CONFIG" ]; then
    cat > "$SSH_CONFIG" << 'EOF'
# GitHub
Host github.com
    HostName github.com
    User git
    IdentitiesOnly yes
    PreferredAuthentications publickey

# GitLab
Host gitlab.com
    HostName gitlab.com
    User git
    IdentitiesOnly yes
    PreferredAuthentications publickey

# Bitbucket
Host bitbucket.org
    HostName bitbucket.org
    User git
    IdentitiesOnly yes
    PreferredAuthentications publickey
EOF
    echo "✅ Created SSH config file with common Git hosts"
else
    echo "✅ SSH config file already exists"
fi

chmod 600 "$SSH_CONFIG"

# Configure Git if not already configured
if ! git config --global user.name >/dev/null 2>&1; then
    echo "⚠️  Git user.name not configured"
    echo "   You may want to set it with: git config --global user.name 'Your Name'"
fi

if ! git config --global user.email >/dev/null 2>&1; then
    echo "⚠️  Git user.email not configured"
    echo "   You may want to set it with: git config --global user.email 'your.email@example.com'"
fi

# Test SSH connection to GitHub (optional)
echo "🔍 Testing SSH connection to GitHub..."
if ssh -T git@github.com -o ConnectTimeout=10 -o StrictHostKeyChecking=no 2>&1 | grep -q "successfully authenticated"; then
    echo "✅ SSH connection to GitHub is working!"
else
    echo "⚠️  Could not establish SSH connection to GitHub"
    echo "   This might be normal if you haven't added your public key to GitHub yet"
fi

echo "🎉 SSH setup completed!"
