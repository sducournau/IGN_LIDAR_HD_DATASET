#!/usr/bin/env python3
"""
Docusaurus Link Validator and Fixer
Analyzes broken links and provides fixes for Docusaurus markdown files.
"""

import re
from pathlib import Path
from collections import defaultdict

class LinkValidator:
    def __init__(self):
        self.docs_dir = Path("docs")
        self.fr_dir = Path("i18n/fr/docusaurus-plugin-content-docs/current")
        self.broken_links = defaultdict(list)
        self.fixes = []
    
    def find_links_in_file(self, file_path):
        """Extract all markdown links from a file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find markdown links: [text](url)
        md_links = re.findall(r'\[([^\]]+)\]\(([^\)]+)\)', content)
        
        return md_links
    
    def validate_link(self, source_file, link_url):
        """Check if a link is valid."""
        # Skip external links
        if link_url.startswith(('http://', 'https://', '#', 'mailto:')):
            return True, None
        
        # Remove anchor
        link_path = link_url.split('#')[0]
        
        # Resolve relative path
        source_dir = source_file.parent
        
        # Handle different link types
        if link_path.startswith('/'):
            # Absolute link from root
            target = self.docs_dir / link_path.lstrip('/')
        elif link_path.startswith('../'):
            # Relative parent
            target = (source_dir / link_path).resolve()
        elif link_path.startswith('./'):
            # Relative current
            target = (source_dir / link_path.lstrip('./')).resolve()
        else:
            # Relative in same dir
            target = source_dir / link_path
        
        # Check if target exists
        if not target.exists():
            return False, str(target)
        
        return True, None
    
    def scan_directory(self, directory):
        """Scan all markdown files in directory."""
        print(f"\n🔍 Scanning: {directory}")
        
        md_files = list(directory.rglob("*.md"))
        broken_count = 0
        
        for md_file in md_files:
            links = self.find_links_in_file(md_file)
            
            for link_text, link_url in links:
                is_valid, broken_path = self.validate_link(md_file, link_url)
                
                if not is_valid:
                    rel_file = md_file.relative_to(directory)
                    self.broken_links[str(rel_file)].append((link_text, link_url, broken_path))
                    broken_count += 1
        
        return broken_count
    
    def suggest_fixes(self):
        """Suggest fixes for broken links."""
        print("\n" + "="*70)
        print("🔧 BROKEN LINKS ANALYSIS AND FIXES")
        print("="*70)
        
        for file, links in sorted(self.broken_links.items()):
            print(f"\n📄 File: {file}")
            print(f"   {len(links)} broken link(s)")
            
            for link_text, link_url, broken_path in links:
                print(f"\n   ❌ Broken: [{link_text}]({link_url})")
                print(f"      Resolved to: {broken_path}")
                
                # Suggest fix
                fix = self.suggest_fix(link_url, broken_path)
                if fix:
                    print(f"      ✅ Suggested fix: {fix}")
                    self.fixes.append({
                        'file': file,
                        'old': link_url,
                        'new': fix,
                        'text': link_text
                    })
    
    def suggest_fix(self, link_url, broken_path):
        """Suggest a fix for a broken link."""
        # Common patterns
        
        # Pattern 1: /docs/ prefix in links (should be removed in Docusaurus)
        if '/docs/' in link_url:
            return link_url.replace('/docs/', '/')
        
        # Pattern 2: .md extension in links (optional in Docusaurus)
        if link_url.endswith('.md'):
            return link_url[:-3]
        
        # Pattern 3: Links to root project files
        if broken_path and broken_path.startswith('/'):
            return f"https://github.com/sducournau/IGN_LIDAR_HD_DATASET/blob/main/{broken_path.lstrip('/')}"
        
        return None
    
    def generate_report(self):
        """Generate a comprehensive report."""
        print("\n" + "="*70)
        print("📊 LINK VALIDATION SUMMARY")
        print("="*70)
        
        total_files = len(self.broken_links)
        total_links = sum(len(links) for links in self.broken_links.values())
        
        print(f"\n📁 Files with broken links: {total_files}")
        print(f"🔗 Total broken links: {total_links}")
        print(f"✅ Suggested fixes: {len(self.fixes)}")
        
        # Categorize issues
        doc_prefix_issues = sum(1 for f in self.fixes if '/docs/' in f['old'])
        md_extension_issues = sum(1 for f in self.fixes if f['old'].endswith('.md'))
        root_file_issues = sum(1 for f in self.fixes if f['old'].startswith('../../../'))
        
        print(f"\n📋 Issue Categories:")
        print(f"   - /docs/ prefix issues: {doc_prefix_issues}")
        print(f"   - .md extension issues: {md_extension_issues}")
        print(f"   - Root file references: {root_file_issues}")
        
        return total_links
    
    def export_fixes(self, filename="link_fixes.txt"):
        """Export fixes to a file."""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# Suggested Link Fixes\n\n")
            
            for fix in self.fixes:
                f.write(f"File: {fix['file']}\n")
                f.write(f"  Old: [{fix['text']}]({fix['old']})\n")
                f.write(f"  New: [{fix['text']}]({fix['new']})\n\n")
        
        print(f"\n💾 Fixes exported to: {filename}")


def main():
    validator = LinkValidator()
    
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║              Docusaurus Link Validator                            ║
╚═══════════════════════════════════════════════════════════════════╝
""")
    
    # Scan English docs
    en_broken = validator.scan_directory(validator.docs_dir)
    
    # Suggest fixes
    if validator.broken_links:
        validator.suggest_fixes()
    
    # Generate report
    validator.generate_report()
    
    # Export fixes
    if validator.fixes:
        validator.export_fixes()
    
    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS:")
    print("="*70)
    print("""
1. Remove /docs/ prefix from internal links
   - Change: [text](/docs/guide) → [text](/guide)
   
2. Remove .md extensions from links (Docusaurus handles this)
   - Change: [text](./file.md) → [text](./file)
   
3. Use proper relative paths
   - Same directory: ./filename
   - Parent directory: ../filename
   - Root: /filename

4. For root project files, link to GitHub
   - Change: ../../../FILE.md → https://github.com/user/repo/blob/main/FILE.md

5. Ensure all referenced files exist in docs/

🔧 TO FIX AUTOMATICALLY:
   Run: python fix_links.py (to be created)

🧪 TO TEST:
   npm run build
""")

if __name__ == "__main__":
    main()
