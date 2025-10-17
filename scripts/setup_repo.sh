#!/bin/bash

# Setup GitHub Repository for CI/CD
# This script helps set up the repository with all necessary configurations

set -e

echo "🔧 Setting up GitHub repository for CI/CD..."

# Create .gitignore if it doesn't exist
if [ ! -f .gitignore ]; then
    echo "📝 Creating .gitignore file..."
    cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*\$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# Environment variables
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Docker
.dockerignore

# Model files (too large for git)
*.pth
*.pt
*.safetensors
models/checkpoints/
!models/checkpoints/.gitkeep

# Data files (too large for git)
data/raw/
data/processed/*.npy
data/processed/*.csv
!data/processed/.gitkeep

# Logs
logs/
*.log

# Temporary files
tmp/
temp/
hf_space_deployment/

# Secrets
secrets.env
.env.local
.env.production
EOF
fi

# Create .dockerignore
echo "🐳 Creating .dockerignore file..."
cat > .dockerignore << EOF
.git
.gitignore
README.md
.env
.venv
venv/
.pytest_cache
.coverage
.nyc_output
node_modules
.DS_Store
*.pyc
*.pyo
*.pyd
__pycache__
.pytest_cache
Dockerfile
.dockerignore
EOF

# Create GitHub issue templates
mkdir -p .github/ISSUE_TEMPLATE

echo "📋 Creating GitHub issue templates..."

cat > .github/ISSUE_TEMPLATE/bug_report.md << EOF
---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment (please complete the following information):**
 - OS: [e.g. iOS]
 - Browser [e.g. chrome, safari]
 - Version [e.g. 22]

**Additional context**
Add any other context about the problem here.
EOF

cat > .github/ISSUE_TEMPLATE/feature_request.md << EOF
---
name: Feature request
about: Suggest an idea for this project
title: '[FEATURE] '
labels: enhancement
assignees: ''
---

**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is. Ex. I'm always frustrated when [...]

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.
EOF

# Create pull request template
echo "🔄 Creating pull request template..."
mkdir -p .github
cat > .github/pull_request_template.md << EOF
## Description

Brief description of the changes in this pull request.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist

- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

## Screenshots (if applicable)

Add screenshots to help explain your changes.
EOF

# Create CONTRIBUTING.md
echo "🤝 Creating CONTRIBUTING.md..."
cat > CONTRIBUTING.md << EOF
# Contributing to E-commerce Sentiment Analysis

Thank you for your interest in contributing to this project! We welcome contributions from everyone.

## Getting Started

1. Fork the repository
2. Clone your fork: \`git clone https://github.com/yourusername/ecommerce_sentiment_agent.git\`
3. Create a new branch: \`git checkout -b feature/your-feature-name\`
4. Set up the development environment (see README.md)

## Development Setup

### Local Development

1. Install dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   pip install -r tests/requirements.txt
   \`\`\`

2. Run tests:
   \`\`\`bash
   python -m pytest tests/ -v
   \`\`\`

3. Run the services locally:
   \`\`\`bash
   docker-compose up --build
   \`\`\`

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for functions and classes
- Keep functions focused and modular

### Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Aim for high test coverage
- Include integration tests where appropriate

## Submitting Changes

1. Commit your changes with clear, descriptive messages
2. Push to your fork: \`git push origin feature/your-feature-name\`
3. Create a Pull Request with a clear description of the changes
4. Ensure CI/CD pipeline passes
5. Address any review feedback

## Reporting Issues

When reporting issues, please include:

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Any relevant logs or error messages

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to create an inclusive and welcoming environment for all contributors.

## Questions?

Feel free to open an issue for any questions about contributing!
EOF

# Create CODE_OF_CONDUCT.md
echo "📜 Creating CODE_OF_CONDUCT.md..."
cat > CODE_OF_CONDUCT.md << EOF
# Code of Conduct

## Our Pledge

We as members, contributors, and leaders pledge to make participation in our
community a harassment-free experience for everyone, regardless of age, body
size, visible or invisible disability, ethnicity, sex characteristics, gender
identity and expression, level of experience, education, socio-economic status,
nationality, personal appearance, race, religion, or sexual identity
and orientation.

## Our Standards

Examples of behavior that contributes to a positive environment include:

* Using welcoming and inclusive language
* Being respectful of differing viewpoints and experiences
* Gracefully accepting constructive criticism
* Focusing on what is best for the community
* Showing empathy towards other community members

Examples of unacceptable behavior include:

* The use of sexualized language or imagery, and sexual attention or advances
* Trolling, insulting or derogatory comments, and personal or political attacks
* Public or private harassment
* Publishing others' private information without explicit permission
* Other conduct which could reasonably be considered inappropriate in a professional setting

## Enforcement

Project maintainers are responsible for clarifying and enforcing our standards of
acceptable behavior and will take appropriate and fair corrective action in
response to any behavior that they deem inappropriate, threatening, offensive,
or harmful.

## Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage],
version 2.0, available at https://www.contributor-covenant.org/version/2/0/code_of_conduct.html

[homepage]: https://www.contributor-covenant.org
EOF

# Make scripts executable
chmod +x scripts/deploy_to_hf.sh

echo "✅ Repository setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Set up GitHub repository secrets:"
echo "   - HF_TOKEN: Your Hugging Face API token"
echo "   - HF_USERNAME: Your Hugging Face username"
echo "   - DOCKER_USERNAME: Your Docker Hub username (optional)"
echo "   - DOCKER_PASSWORD: Your Docker Hub password (optional)"
echo ""
echo "2. Initialize git repository (if not already done):"
echo "   git init"
echo "   git add ."
echo "   git commit -m 'Initial commit'"
echo "   git branch -M main"
echo "   git remote add origin <your-repo-url>"
echo "   git push -u origin main"
echo ""
echo "3. Enable GitHub Actions in your repository"
echo ""
echo "4. Test deployment with:"
echo "   ./scripts/deploy_to_hf.sh"
EOF