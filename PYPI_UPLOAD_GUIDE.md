# PyPI Upload Guide for IGN LiDAR HD Processing Library

## Package Ready for Upload ✅

Your package has been successfully prepared for PyPI upload. All required files have been built and validated.

## Built Distribution Files

The following distribution files are ready in the `dist/` directory:

- `ign_lidar_hd-1.1.0-py3-none-any.whl` (wheel format)
- `ign_lidar_hd-1.1.0.tar.gz` (source distribution)

Both files have passed twine validation checks.

## Uploading to PyPI

### Test Upload (Recommended First Step)

1. **Create TestPyPI account** (if you don't have one):

   - Go to [test.pypi.org](https://test.pypi.org/)
   - Register for an account

2. **Upload to TestPyPI**:

   ```bash
   source .venv/bin/activate
   twine upload --repository testpypi dist/*
   ```

3. **Test installation from TestPyPI**:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ ign-lidar-hd
   ```

### Production Upload

1. **Create PyPI account** (if you don't have one):

   - Go to [pypi.org](https://pypi.org/)
   - Register for an account

2. **Upload to PyPI**:
   ```bash
   source .venv/bin/activate
   twine upload dist/*
   ```

### Using API Tokens (Recommended)

Instead of using username/password, create API tokens:

1. Go to your PyPI account settings
2. Create a new API token
3. Use the token for upload:
   ```bash
   twine upload --username __token__ --password your-token-here dist/*
   ```

## GitHub Repository Description

### Current Repository Information

- **Repository**: https://github.com/sducournau/IGN_LIDAR_HD_DATASET
- **Package Name**: ign-lidar-hd (PyPI)

### Recommended GitHub Repository Settings

#### Repository Description

Add this description to your GitHub repository:

```
🏗️ Comprehensive Python library for processing IGN LiDAR HD data into machine learning-ready datasets for Building Level of Detail (LOD) classification. Features GPU/CPU processing, smart data management, and complete ML pipeline integration.
```

#### Repository Topics/Tags

Add these tags to improve discoverability:

```
lidar, machine-learning, gis, building-classification, ign, point-cloud, france, geospatial, computer-vision, data-processing, pytorch, numpy, scikit-learn
```

#### About Section

- **Website**: Leave empty or add documentation URL when available
- **Topics**: Use the tags listed above
- **Include in the home page**: Check this box
- **Packages**: Should automatically detect the published PyPI package

## Post-Upload Steps

### 1. Update README Badges

Your README already has a PyPI badge. After upload, verify it works:

```markdown
[![PyPI version](https://badge.fury.io/py/ign-lidar-hd.svg)](https://badge.fury.io/py/ign-lidar-hd)
```

### 2. Create GitHub Release

1. Go to your repository releases
2. Create a new release with tag `v1.1.0`
3. Use the changelog content for release notes
4. Attach the distribution files (optional)

### 3. Documentation Updates

- Update any installation instructions to use `pip install ign-lidar-hd`
- Ensure examples work with the installed package

## Package Management

### Version Updates

When updating the package:

1. **Update version** in `pyproject.toml`:

   ```toml
   version = "1.1.1"  # or next version
   ```

2. **Clean and rebuild**:

   ```bash
   rm -rf dist/ build/ ign_lidar_hd.egg-info/
   python -m build
   ```

3. **Validate and upload**:
   ```bash
   twine check dist/*
   twine upload dist/*
   ```

### Development Dependencies

The package includes development dependencies in `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=6.0",
    "pytest-cov>=4.0",
    "black>=22.0",
    "flake8>=4.0",
    "mypy>=0.910",
    "build>=0.7.0",
    "twine>=4.0",
    "pre-commit>=2.20",
]
```

Install for development:

```bash
pip install -e ".[dev]"
```

## Troubleshooting

### Common Issues

1. **Authentication Error**: Use API tokens instead of username/password
2. **Package Name Conflict**: The name `ign-lidar-hd` should be available
3. **File Size Limits**: PyPI has size limits; your package is well within them
4. **Metadata Issues**: All metadata has been validated with twine

### Support

- PyPI Documentation: https://packaging.python.org/
- Twine Documentation: https://twine.readthedocs.io/
- Setuptools Documentation: https://setuptools.pypa.io/

## Security Notes

- Never commit PyPI tokens to your repository
- Use environment variables or keyring for token storage
- Consider using GitHub Actions for automated publishing
- Enable 2FA on your PyPI account

## Next Steps

1. ✅ Package built and validated
2. 🔄 Upload to TestPyPI (recommended)
3. 🔄 Test installation from TestPyPI
4. 🔄 Upload to production PyPI
5. 🔄 Update GitHub repository description and tags
6. 🔄 Create GitHub release v1.1.0
