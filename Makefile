# ============================================================
#  EPOQ — Developer Makefile
#  Run `make help` to see all available commands
# ============================================================

.PHONY: help install dev build clean

# Default target
help:
	@echo ""
	@echo "  EPOQ — Available Commands"
	@echo "  ========================="
	@echo "  make install         Install Node.js dependencies"
	@echo "  make python-install  Install Python ML dependencies"
	@echo "  make setup           Run full setup (Node + Python)"
	@echo "  make dev             Start the development server"
	@echo "  make build           Build the production desktop app"
	@echo "  make lint            Run the frontend linter"
	@echo "  make clean           Remove build artifacts"
	@echo ""

install:
	@echo "📦 Installing Node dependencies..."
	cd image-trainer && npm install

dev:
	@echo "🚀 Starting EPOQ development server..."
	cd image-trainer && npm run tauri dev

build:
	@echo "🔨 Building EPOQ for production..."
	cd image-trainer && npm run tauri build

clean:
	@echo "🧹 Cleaning build artifacts..."
	cd image-trainer && rm -rf .next node_modules src-tauri/target
	@echo "✅ Clean complete."

python-install:
	@echo "🐍 Installing Python ML dependencies..."
	pip install torch torchvision pandas scikit-learn matplotlib seaborn
	@echo "✅ Python dependencies installed."

lint:
	@echo "🔍 Running linter..."
	cd image-trainer && npm run lint

setup: install python-install
	@echo "✅ Full setup complete! Run 'make dev' to start."