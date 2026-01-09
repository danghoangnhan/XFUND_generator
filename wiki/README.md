# Wiki Content

This directory contains the GitHub Wiki pages for XFUND Generator.

## Deploying to GitHub Wiki

### Option 1: Manual Copy

1. Go to the repository wiki: https://github.com/danghoangnhan/XFUND_generator/wiki
2. Create pages manually and paste content from each `.md` file

### Option 2: Git Clone (Recommended)

```bash
# Clone the wiki repository
git clone https://github.com/danghoangnhan/XFUND_generator.wiki.git

# Copy wiki files
cp wiki/*.md XFUND_generator.wiki/

# Commit and push
cd XFUND_generator.wiki
git add .
git commit -m "Update wiki documentation"
git push
```

## Wiki Pages

| File | Description |
|------|-------------|
| `Home.md` | Main wiki page |
| `Installation.md` | Installation guide |
| `Getting-Started.md` | Getting started guide |
| `Configuration.md` | Configuration reference |
| `Annotation-Formats.md` | XFUND/FUNSD/WildReceipt formats |
| `API-Reference.md` | Python API documentation |
| `Testing.md` | Testing guide |
| `Contributing.md` | Contribution guidelines |
| `_Sidebar.md` | Wiki sidebar navigation |

## Updating Wiki

When documentation changes:

1. Update files in this `wiki/` directory
2. Deploy to GitHub Wiki using the method above
