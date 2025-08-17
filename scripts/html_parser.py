from pathlib import Path
from typing import List
from bs4 import BeautifulSoup, NavigableString, Tag
import re


class HTMLParser:
    """Enhanced HTML Parser for extracting structured content from HTML files
    
    Converts HTML documents to clean markdown format, preserving tables, lists,
    and document structure for better RAG system processing.
    """
    
    def __init__(self, output_dir: str = "data/processed"):
        """Initialize HTMLParser
        
        Args:
            output_dir: Directory to save processed files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _table_to_markdown(self, table: Tag) -> str:
        """Convert HTML table to markdown format
        
        Args:
            table: BeautifulSoup table element
            
        Returns:
            Markdown formatted table string
        """
        markdown_lines = []
        
        # Extract all rows
        rows = table.find_all('tr')
        if not rows:
            return ""
        
        # Process each row
        table_data = []
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if cells:
                row_data = []
                for cell in cells:
                    # Get cell text and clean it
                    cell_text = cell.get_text(strip=True)
                    # Replace newlines with spaces for table cells
                    cell_text = ' '.join(cell_text.split())
                    # Escape pipe characters if present
                    cell_text = cell_text.replace('|', '\\|')
                    row_data.append(cell_text)
                table_data.append(row_data)
        
        if not table_data:
            return ""
        
        # Determine if first row is header (contains th elements or is first row)
        first_row_has_th = bool(rows[0].find_all('th'))
        
        # Build markdown table
        for i, row in enumerate(table_data):
            # Add pipes and content
            markdown_lines.append('| ' + ' | '.join(row) + ' |')
            
            # Add separator after header row
            if i == 0 and (first_row_has_th or len(table_data) > 1):
                separator = '|' + '|'.join([' --- ' for _ in row]) + '|'
                markdown_lines.append(separator)
        
        return '\n'.join(markdown_lines)
    
    def _process_list(self, list_elem: Tag, list_type: str = 'ul') -> str:
        """Convert HTML list to markdown format
        
        Args:
            list_elem: BeautifulSoup list element (ul or ol)
            list_type: Type of list ('ul' or 'ol')
            
        Returns:
            Markdown formatted list string
        """
        markdown_lines = []
        items = list_elem.find_all('li', recursive=False)
        
        for i, item in enumerate(items):
            # Get item text
            item_text = item.get_text(strip=True)
            
            # Format based on list type
            if list_type == 'ol':
                prefix = f"{i + 1}. "
            else:
                prefix = "- "
            
            # Handle nested lists if present
            nested_lists = item.find_all(['ul', 'ol'], recursive=False)
            if nested_lists:
                # Get text before nested list
                for nested in nested_lists:
                    nested.extract()
                item_text = item.get_text(strip=True)
            
            markdown_lines.append(prefix + item_text)
            
            # Add nested lists with indentation
            if nested_lists:
                for nested in nested_lists:
                    nested_type = nested.name
                    nested_markdown = self._process_list(nested, nested_type)
                    # Indent nested list items
                    nested_lines = nested_markdown.split('\n')
                    indented_lines = ['  ' + line for line in nested_lines if line]
                    markdown_lines.extend(indented_lines)
        
        return '\n'.join(markdown_lines)
    
    def _process_element(self, element: Tag) -> str:
        """Process a single HTML element and convert to markdown
        
        Args:
            element: BeautifulSoup element
            
        Returns:
            Markdown formatted string
        """
        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            # Convert headers to markdown
            level = int(element.name[1])
            header_text = element.get_text(strip=True)
            return '#' * level + ' ' + header_text + '\n'
        
        elif element.name == 'table':
            # Convert table to markdown
            return self._table_to_markdown(element) + '\n'
        
        elif element.name == 'ul':
            # Convert unordered list to markdown
            return self._process_list(element, 'ul') + '\n'
        
        elif element.name == 'ol':
            # Convert ordered list to markdown
            return self._process_list(element, 'ol') + '\n'
        
        elif element.name == 'p':
            # Paragraphs
            text = element.get_text(strip=True)
            if text:
                return text + '\n\n'
            return ''
        
        elif element.name == 'blockquote':
            # Blockquotes
            text = element.get_text(strip=True)
            if text:
                lines = text.split('\n')
                quoted_lines = ['> ' + line for line in lines]
                return '\n'.join(quoted_lines) + '\n\n'
            return ''
        
        elif element.name == 'hr':
            # Horizontal rule
            return '---\n\n'
        
        elif element.name == 'br':
            # Line break
            return '\n'
        
        elif element.name in ['div', 'section', 'article', 'main']:
            # Container elements - process their children
            return ''
        
        else:
            # For other elements, just get text if it's meaningful
            text = element.get_text(strip=True)
            if text and element.name not in ['script', 'style', 'meta', 'link']:
                return text + '\n'
            return ''
    
    def extract_structured_content(self, html_file: str) -> str:
        """Extract structured content from HTML file, preserving tables and formatting
        
        Args:
            html_file: Path to the HTML file
            
        Returns:
            str: Markdown content
        """
        try:
            # Read HTML file
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Parse HTML content with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title if present
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text(strip=True)
            
            # Remove script and style elements
            for script in soup(["script", "style", "meta", "link"]):
                script.decompose()
            
            # Find the main content area (body or main content div)
            body = soup.find('body')
            if not body:
                body = soup
            
            # Process content recursively
            markdown_content = []
            
            def process_children(parent):
                """Recursively process child elements"""
                for child in parent.children:
                    if isinstance(child, NavigableString):
                        # Handle text nodes
                        text = str(child).strip()
                        if text:
                            markdown_content.append(text)
                    elif isinstance(child, Tag):
                        # Process the element
                        result = self._process_element(child)
                        if result:
                            markdown_content.append(result)
                        # If it's a container element, process its children
                        elif child.name in ['div', 'section', 'article', 'main', 'body']:
                            process_children(child)
            
            # Start processing from body
            process_children(body)
            
            # Join all markdown content
            final_content = '\n'.join(markdown_content)
            
            # Clean up multiple newlines
            final_content = re.sub(r'\n{3,}', '\n\n', final_content)
            
            # Add title at the beginning if present
            if title:
                final_content = f"# {title}\n\n{final_content}"
            
            return final_content
            
        except Exception as e:
            print(f"Error processing {html_file}: {e}")
            return ""
    
    def save_processed_data(self, content: str, base_filename: str) -> None:
        """Save processed content to markdown file
        
        Args:
            content: Extracted markdown content
            base_filename: Base filename (without extension)
        """
        try:
            # Save markdown content
            markdown_file = self.output_dir / f"{base_filename}.md"
            with open(markdown_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
        except Exception as e:
            print(f"Error saving processed data for {base_filename}: {e}")
    
    def process_html_files(self, input_dir: str) -> List[Dict[str, Any]]:
        """Process all HTML files in a directory
        
        Args:
            input_dir: Directory containing HTML files
            
        Returns:
            List[Dict[str, Any]]: List of processing results
        """
        input_path = Path(input_dir)
        results = []
        
        if not input_path.exists():
            print(f"Input directory {input_dir} does not exist")
            return results
        
        # Find all HTML files
        html_files = list(input_path.glob("*.html")) + list(input_path.glob("*.htm"))
        
        if not html_files:
            print(f"No HTML files found in {input_dir}")
            return results
        
        print(f"Found {len(html_files)} HTML files to process")
        print("-" * 40)
        
        # Process each HTML file
        for i, html_file in enumerate(html_files, 1):
            print(f"[{i}/{len(html_files)}] Processing: {html_file.name}")
            
            # Extract structured content
            content = self.extract_structured_content(str(html_file))
            
            if content:  # Only save if extraction was successful
                # Get base filename without extension
                base_filename = html_file.stem
                
                # Save processed data
                self.save_processed_data(content, base_filename)
                
                # Add to results
                result = {
                    "source_file": str(html_file),
                    "base_filename": base_filename,
                    "success": True,
                    "content_length": len(content),
                    "word_count": len(content.split()),
                }
                results.append(result)
                
                # Print summary for this file
                print(f"  ✓ Extracted: {result['word_count']} words")
            else:
                # Record failed processing
                result = {
                    "source_file": str(html_file),
                    "base_filename": html_file.stem,
                    "success": False,
                    "error": "Failed to extract content"
                }
                results.append(result)
                print(f"  ✗ Failed to extract content")
        
        return results


def main():
    """Main function to process HTML files from data/raw directory"""
    # Set up paths
    input_dir = "data/raw"
    output_dir = "data/processed"
    
    print("Enhanced HTML Parser - Structured Content Extraction")
    print("=" * 50)
    
    # Initialize parser
    parser = HTMLParser(output_dir)
    
    # Process HTML files
    results = parser.process_html_files(input_dir)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Processing Summary:")
    print("-" * 20)
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"Total files processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        total_words = sum(r.get('word_count', 0) for r in results if r['success'])
        
        print(f"\nContent Statistics:")
        print(f"  Total words extracted: {total_words:}")
    
    print(f"\nProcessing complete!")
    print(f"Markdown files saved to: {output_dir}")


if __name__ == "__main__":
    main()