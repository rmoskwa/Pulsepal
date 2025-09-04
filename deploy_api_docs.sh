#!/bin/bash
# Deploy only MATLAB API documentation to GitHub Pages

echo "Creating API-only documentation site..."

# Create temporary directory for API docs
rm -rf api-docs-deploy
mkdir -p api-docs-deploy/docs

# Copy the MATLAB API documentation and sequences
cp -r docs/matlab_api api-docs-deploy/docs/
cp -r docs/sequences api-docs-deploy/docs/
cp docs/index.md api-docs-deploy/docs/
cp mkdocs.yml api-docs-deploy/

# Navigate to deployment directory
cd api-docs-deploy

echo "Building and deploying to GitHub Pages..."
mkdocs gh-deploy --force

cd ..
echo "Deployment complete! Your API docs will be available at:"
echo "https://[your-username].github.io/pulsePal/"
echo ""
echo "Note: It may take 2-5 minutes for the site to be live."
