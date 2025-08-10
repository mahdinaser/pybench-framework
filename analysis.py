#!/usr/bin/env python3
"""
PyBench Report Table Generator

Reads PyBench JSON report files and generates comprehensive tables in multiple formats:
- HTML tables with styling
- CSV files for data analysis
- Console output with formatting
- Excel workbook with multiple sheets

Usage:
    python table_generator.py report.json
    python table_generator.py report.json --format html csv console
    python table_generator.py report.json --output my_tables
"""

import json
import csv
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import statistics
from datetime import datetime

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Note: pandas not available. Excel output will be disabled.")

class PyBenchTableGenerator:
    """Generate tables from PyBench JSON reports"""
    
    def __init__(self, report_file: str):
        self.report_file = Path(report_file)
        self.data = self._load_report()
        self.output_dir = self.report_file.parent
        
    def _load_report(self) -> Dict[str, Any]:
        """Load and validate PyBench JSON report"""
        try:
            with open(self.report_file, 'r') as f:
                data = json.load(f)
            
            # Validate required structure
            required_keys = ['study_metadata', 'results', 'reproducibility']
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Invalid PyBench report: missing '{key}' section")
            
            print(f"✅ Loaded PyBench report: {self.report_file}")
            print(f"   • Total experiments: {data['study_metadata']['total_experiments']}")
            print(f"   • Date: {data['study_metadata']['date']}")
            
            return data
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Report file not found: {self.report_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in report file: {e}")
    
    def generate_all_tables(self, formats: List[str] = None, output_prefix: str = None) -> Dict[str, str]:
        """Generate all tables in specified formats"""
        if formats is None:
            formats = ['console', 'html', 'csv']
        
        if output_prefix is None:
            output_prefix = self.report_file.stem + "_tables"
        
        generated_files = {}
        
        # Generate each table type
        tables = {
            'platform_availability': self._generate_platform_availability_table(),
            'performance_rankings': self._generate_performance_rankings_tables(),
            'detailed_metrics': self._generate_detailed_metrics_table(),
            'category_summary': self._generate_category_summary_table(),
            'statistical_summary': self._generate_statistical_summary_table(),
            'system_info': self._generate_system_info_table()
        }
        
        # Output in requested formats
        if 'console' in formats:
            self._output_console_tables(tables)
        
        if 'html' in formats:
            html_file = self._output_html_tables(tables, output_prefix)
            generated_files['html'] = html_file
        
        if 'csv' in formats:
            csv_files = self._output_csv_tables(tables, output_prefix)
            generated_files['csv'] = csv_files
        
        if 'excel' in formats and PANDAS_AVAILABLE:
            excel_file = self._output_excel_tables(tables, output_prefix)
            generated_files['excel'] = excel_file
        
        return generated_files
    
    def _generate_platform_availability_table(self) -> List[Dict[str, Any]]:
        """Generate platform availability summary table"""
        table = []
        platform_status = self.data['results']['platform_status']
        
        # Get platform definitions if available
        platforms_info = {}
        if 'platforms' in self.data['results']:
            platforms_info = self.data['results']['platforms']
        
        for platform_id, status_info in platform_status.items():
            row = {
                'Platform ID': platform_id,
                'Status': '✅ Available' if status_info['status'] == 'available' else '❌ Not Available',
                'Category': status_info.get('category', 'Unknown'),
                'Subcategory': status_info.get('subcategory', 'Unknown'),
                'Version': status_info.get('version', 'Unknown'),
                'Import Path': status_info.get('import_path', 'Unknown'),
                'Error': status_info.get('error', '') if status_info['status'] != 'available' else ''
            }
            
            # Add platform info if available
            if platform_id in platforms_info:
                info = platforms_info[platform_id]
                row.update({
                    'Name': info.get('name', platform_id),
                    'Emoji': info.get('emoji', ''),
                    'Complexity Class': info.get('complexity_class', 'Unknown'),
                    'Memory Model': info.get('memory_model', 'Unknown')
                })
            
            table.append(row)
        
        # Sort by category, then by platform name
        table.sort(key=lambda x: (x['Category'], x['Platform ID']))
        return table
    
    def _generate_performance_rankings_tables(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate performance rankings tables for each operation"""
        rankings_tables = {}
        performance_rankings = self.data['results'].get('performance_rankings', {})
        
        for operation, rankings in performance_rankings.items():
            table = []
            for i, rank_info in enumerate(rankings, 1):
                row = {
                    'Rank': i,
                    'Library': rank_info['library'],
                    'Geometric Mean (s)': f"{rank_info['geometric_mean']:.6f}",
                    'Std Dev': f"{rank_info['std_dev']:.6f}",
                    'Coefficient of Variation': f"{rank_info['coefficient_variation']:.4f}",
                    'Measurements': rank_info['measurements'],
                    'Performance Status': self._get_performance_status(rank_info['coefficient_variation'])
                }
                table.append(row)
            
            rankings_tables[f"{operation}_rankings"] = table
        
        return rankings_tables
    
    def _generate_detailed_metrics_table(self) -> List[Dict[str, Any]]:
        """Generate detailed performance metrics table"""
        table = []
        raw_measurements = self.data['results'].get('raw_measurements', [])
        
        for measurement in raw_measurements:
            execution_times = measurement['execution_times']
            if execution_times:
                row = {
                    'Library': measurement['library'],
                    'Operation': measurement['operation'],
                    'Data Size': f"{measurement['data_size']:,}",
                    'Mean Time (s)': f"{statistics.mean(execution_times):.6f}",
                    'Median Time (s)': f"{statistics.median(execution_times):.6f}",
                    'Std Dev (s)': f"{statistics.stdev(execution_times) if len(execution_times) > 1 else 0:.6f}",
                    'Min Time (s)': f"{min(execution_times):.6f}",
                    'Max Time (s)': f"{max(execution_times):.6f}",
                    'Memory Peak (MB)': f"{measurement.get('memory_peak_mb', 0):.2f}",
                    'Memory Mean (MB)': f"{measurement.get('memory_mean_mb', 0):.2f}",
                    'CPU Utilization (%)': f"{measurement.get('cpu_utilization', 0):.2f}"
                }
                table.append(row)
        
        # Sort by library, then operation, then data size
        table.sort(key=lambda x: (x['Library'], x['Operation'], int(x['Data Size'].replace(',', ''))))
        return table
    
    def _generate_category_summary_table(self) -> List[Dict[str, Any]]:
        """Generate category-wise summary table"""
        table = []
        platform_status = self.data['results']['platform_status']
        
        # Group by category
        category_stats = defaultdict(lambda: {
            'total': 0, 
            'available': 0, 
            'platforms': [],
            'best_performer': None,
            'best_time': float('inf')
        })
        
        for platform_id, status_info in platform_status.items():
            category = status_info.get('category', 'Unknown')
            category_stats[category]['total'] += 1
            category_stats[category]['platforms'].append(platform_id)
            
            if status_info['status'] == 'available':
                category_stats[category]['available'] += 1
        
        # Find best performers
        raw_measurements = self.data['results'].get('raw_measurements', [])
        for measurement in raw_measurements:
            platform_id = measurement['library']
            if platform_id in platform_status:
                category = platform_status[platform_id].get('category', 'Unknown')
                mean_time = statistics.mean(measurement['execution_times']) if measurement['execution_times'] else float('inf')
                
                if mean_time < category_stats[category]['best_time']:
                    category_stats[category]['best_time'] = mean_time
                    category_stats[category]['best_performer'] = platform_id
        
        # Build table
        for category, stats in sorted(category_stats.items()):
            coverage_pct = (stats['available'] / stats['total'] * 100) if stats['total'] > 0 else 0
            
            row = {
                'Category': category,
                'Total Platforms': stats['total'],
                'Available': stats['available'],
                'Coverage (%)': f"{coverage_pct:.1f}%",
                'Best Performer': stats['best_performer'] or 'N/A',
                'Best Time (s)': f"{stats['best_time']:.6f}" if stats['best_time'] != float('inf') else 'N/A',
                'Platforms': ', '.join(stats['platforms'][:5]) + ('...' if len(stats['platforms']) > 5 else '')
            }
            table.append(row)
        
        return table
    
    def _generate_statistical_summary_table(self) -> List[Dict[str, Any]]:
        """Generate statistical analysis summary table"""
        table = []
        statistical_summary = self.data['results'].get('statistical_summary', {})
        
        for operation, summary in statistical_summary.items():
            comparisons = summary.get('pairwise_comparisons', [])
            total_comparisons = len(comparisons)
            significant_comparisons = sum(1 for c in comparisons if c.get('significant', False))
            
            row = {
                'Operation': operation.capitalize(),
                'Libraries Tested': summary.get('libraries_tested', 0),
                'Total Measurements': summary.get('measurements_count', 0),
                'Pairwise Comparisons': total_comparisons,
                'Significant Differences': significant_comparisons,
                'Significance Rate (%)': f"{(significant_comparisons / total_comparisons * 100) if total_comparisons > 0 else 0:.1f}%"
            }
            table.append(row)
        
        return table
    
    def _generate_system_info_table(self) -> List[Dict[str, Any]]:
        """Generate system information table"""
        table = []
        metadata = self.data['results']['metadata']
        config = self.data['reproducibility']['configuration']
        
        system_info = [
            {'Property': 'Framework Version', 'Value': self.data['study_metadata'].get('framework', 'PyBench v1.0')},
            {'Property': 'Report Date', 'Value': self.data['study_metadata'].get('date', 'Unknown')},
            {'Property': 'Python Version', 'Value': metadata.get('python_version', 'Unknown').split()[0]},
            {'Property': 'Platform', 'Value': metadata.get('platform', 'Unknown')},
            {'Property': 'CPU Cores (Logical)', 'Value': str(metadata.get('cpu_count_logical', 'Unknown'))},
            {'Property': 'CPU Cores (Physical)', 'Value': str(metadata.get('cpu_count_physical', 'Unknown'))},
            {'Property': 'Memory (GB)', 'Value': f"{metadata.get('memory_total_gb', 0):.1f}"},
            {'Property': 'Random Seed', 'Value': str(config.get('random_seed', 'Unknown'))},
            {'Property': 'Trials per Measurement', 'Value': str(config.get('num_trials', 'Unknown'))},
            {'Property': 'Confidence Level', 'Value': f"{config.get('confidence_level', 0.95)*100:.0f}%"},
            {'Property': 'Total Experiments', 'Value': f"{self.data['study_metadata'].get('total_experiments', 0):,}"},
            {'Property': 'Total Platforms', 'Value': str(self.data['study_metadata'].get('total_platforms', 0))},
            {'Property': 'Available Platforms', 'Value': str(self.data['study_metadata'].get('available_platforms', 0))}
        ]
        
        return system_info
    
    def _get_performance_status(self, cv: float) -> str:
        """Get performance status based on coefficient of variation"""
        if cv < 0.1:
            return "🟢 Stable"
        elif cv < 0.3:
            return "🟡 Variable"
        else:
            return "🔴 Unstable"
    
    def _output_console_tables(self, tables: Dict[str, Any]) -> None:
        """Output tables to console with formatting"""
        print("\n" + "="*120)
        print("📊 PYBENCH PERFORMANCE ANALYSIS TABLES")
        print("="*120)
        
        # Platform Availability Table
        print(f"\n📋 TABLE 1: PLATFORM AVAILABILITY STATUS")
        print("-"*120)
        self._print_table_console(tables['platform_availability'], max_rows=20)
        
        # Performance Rankings Tables
        rankings = tables['performance_rankings']
        for i, (operation_key, ranking_table) in enumerate(rankings.items(), 2):
            operation = operation_key.replace('_rankings', '').upper()
            print(f"\n🏆 TABLE {i}: PERFORMANCE RANKINGS - {operation} OPERATION")
            print("-"*120)
            self._print_table_console(ranking_table, max_rows=10)
        
        # Category Summary
        table_num = len(rankings) + 2
        print(f"\n📈 TABLE {table_num}: CATEGORY SUMMARY")
        print("-"*120)
        self._print_table_console(tables['category_summary'])
        
        # Statistical Summary
        table_num += 1
        print(f"\n📊 TABLE {table_num}: STATISTICAL ANALYSIS SUMMARY")
        print("-"*120)
        self._print_table_console(tables['statistical_summary'])
        
        # System Information
        table_num += 1
        print(f"\n💻 TABLE {table_num}: SYSTEM INFORMATION")
        print("-"*120)
        self._print_table_console(tables['system_info'])
        
        print(f"\n📋 DETAILED METRICS (First 20 rows)")
        print("-"*120)
        self._print_table_console(tables['detailed_metrics'], max_rows=20)
        
        print("\n" + "="*120)
    
    def _print_table_console(self, table: List[Dict[str, Any]], max_rows: int = None) -> None:
        """Print table to console with proper formatting"""
        if not table:
            print("   No data available")
            return
        
        # Limit rows if specified
        display_table = table[:max_rows] if max_rows else table
        
        # Get column widths
        headers = list(display_table[0].keys())
        col_widths = {}
        for header in headers:
            col_widths[header] = max(len(header), max(len(str(row[header])) for row in display_table))
            col_widths[header] = min(col_widths[header], 30)  # Max width
        
        # Print header
        header_row = " | ".join(header.ljust(col_widths[header]) for header in headers)
        print(header_row)
        print("-" * len(header_row))
        
        # Print rows
        for row in display_table:
            data_row = " | ".join(str(row[header])[:col_widths[header]].ljust(col_widths[header]) for header in headers)
            print(data_row)
        
        if max_rows and len(table) > max_rows:
            print(f"... and {len(table) - max_rows} more rows")
    
    def _output_html_tables(self, tables: Dict[str, Any], output_prefix: str) -> str:
        """Output tables as HTML file"""
        html_file = self.output_dir / f"{output_prefix}.html"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PyBench Performance Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; background: white; }}
        th {{ background: linear-gradient(135deg, #3498db, #2980b9); color: white; padding: 12px; text-align: left; border: 1px solid #ddd; }}
        td {{ padding: 8px 12px; border: 1px solid #ddd; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        tr:hover {{ background-color: #e3f2fd; }}
        .metadata {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .performance-stable {{ color: #27ae60; font-weight: bold; }}
        .performance-variable {{ color: #f39c12; font-weight: bold; }}
        .performance-unstable {{ color: #e74c3c; font-weight: bold; }}
        .available {{ color: #27ae60; font-weight: bold; }}
        .not-available {{ color: #e74c3c; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 PyBench Performance Analysis Report</h1>
        
        <div class="metadata">
            <h3>📊 Report Metadata</h3>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Source:</strong> {self.report_file.name}</p>
            <p><strong>Total Experiments:</strong> {self.data['study_metadata'].get('total_experiments', 0):,}</p>
            <p><strong>Framework:</strong> {self.data['study_metadata'].get('framework', 'PyBench v1.0')}</p>
        </div>
"""
        
        # Platform Availability Table
        html_content += "<h2>📋 Platform Availability Status</h2>\n"
        html_content += self._table_to_html(tables['platform_availability'], 'platform-availability')
        
        # Performance Rankings Tables
        rankings = tables['performance_rankings']
        for operation_key, ranking_table in rankings.items():
            operation = operation_key.replace('_rankings', '').upper()
            html_content += f"<h2>🏆 Performance Rankings - {operation} Operation</h2>\n"
            html_content += self._table_to_html(ranking_table, f'rankings-{operation.lower()}')
        
        # Category Summary
        html_content += "<h2>📈 Category Summary</h2>\n"
        html_content += self._table_to_html(tables['category_summary'], 'category-summary')
        
        # Statistical Summary
        html_content += "<h2>📊 Statistical Analysis Summary</h2>\n"
        html_content += self._table_to_html(tables['statistical_summary'], 'statistical-summary')
        
        # System Information
        html_content += "<h2>💻 System Information</h2>\n"
        html_content += self._table_to_html(tables['system_info'], 'system-info')
        
        # Detailed Metrics (first 100 rows)
        html_content += "<h2>📋 Detailed Performance Metrics (Top 100)</h2>\n"
        html_content += self._table_to_html(tables['detailed_metrics'][:100], 'detailed-metrics')
        
        html_content += """
    </div>
</body>
</html>
"""
        
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"📄 HTML report saved: {html_file}")
        return str(html_file)
    
    def _table_to_html(self, table: List[Dict[str, Any]], table_id: str) -> str:
        """Convert table to HTML"""
        if not table:
            return "<p>No data available</p>\n"
        
        html = f'<table id="{table_id}">\n<thead>\n<tr>\n'
        
        # Headers
        for header in table[0].keys():
            html += f'<th>{header}</th>\n'
        html += '</tr>\n</thead>\n<tbody>\n'
        
        # Rows
        for row in table:
            html += '<tr>\n'
            for header, value in row.items():
                # Apply styling based on content
                css_class = ""
                if "Status" in header and "Available" in str(value):
                    css_class = "available" if "✅" in str(value) else "not-available"
                elif "Performance Status" in header:
                    if "Stable" in str(value):
                        css_class = "performance-stable"
                    elif "Variable" in str(value):
                        css_class = "performance-variable"
                    elif "Unstable" in str(value):
                        css_class = "performance-unstable"
                
                html += f'<td class="{css_class}">{value}</td>\n'
            html += '</tr>\n'
        
        html += '</tbody>\n</table>\n'
        return html
    
    def _output_csv_tables(self, tables: Dict[str, Any], output_prefix: str) -> List[str]:
        """Output tables as CSV files"""
        csv_files = []
        
        # Single tables
        single_tables = {
            'platform_availability': tables['platform_availability'],
            'category_summary': tables['category_summary'],
            'statistical_summary': tables['statistical_summary'],
            'system_info': tables['system_info'],
            'detailed_metrics': tables['detailed_metrics']
        }
        
        for table_name, table_data in single_tables.items():
            csv_file = self.output_dir / f"{output_prefix}_{table_name}.csv"
            self._write_csv(table_data, csv_file)
            csv_files.append(str(csv_file))
        
        # Performance rankings tables
        rankings = tables['performance_rankings']
        for operation_key, ranking_table in rankings.items():
            csv_file = self.output_dir / f"{output_prefix}_{operation_key}.csv"
            self._write_csv(ranking_table, csv_file)
            csv_files.append(str(csv_file))
        
        print(f"📊 CSV files saved: {len(csv_files)} files")
        return csv_files
    
    def _write_csv(self, table: List[Dict[str, Any]], filename: Path) -> None:
        """Write table to CSV file"""
        if not table:
            return
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=table[0].keys())
            writer.writeheader()
            writer.writerows(table)
    
    def _output_excel_tables(self, tables: Dict[str, Any], output_prefix: str) -> str:
        """Output tables as Excel workbook with multiple sheets"""
        if not PANDAS_AVAILABLE:
            print("⚠️  Pandas not available, skipping Excel output")
            return ""
        
        excel_file = self.output_dir / f"{output_prefix}.xlsx"
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Single tables
            single_tables = {
                'Platform_Availability': tables['platform_availability'],
                'Category_Summary': tables['category_summary'],
                'Statistical_Summary': tables['statistical_summary'],
                'System_Info': tables['system_info'],
                'Detailed_Metrics': tables['detailed_metrics']
            }
            
            for sheet_name, table_data in single_tables.items():
                if table_data:
                    df = pd.DataFrame(table_data)
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Performance rankings sheets
            rankings = tables['performance_rankings']
            for operation_key, ranking_table in rankings.items():
                if ranking_table:
                    sheet_name = operation_key.replace('_rankings', '').capitalize() + '_Rankings'
                    df = pd.DataFrame(ranking_table)
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"📈 Excel workbook saved: {excel_file}")
        return str(excel_file)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate tables from PyBench JSON reports'
    )
    parser.add_argument('report_file', help='Path to PyBench JSON report file')
    parser.add_argument('--format', nargs='+', 
                       choices=['console', 'html', 'csv', 'excel'],
                       default=['console', 'html', 'csv'],
                       help='Output formats (default: console html csv)')
    parser.add_argument('--output', help='Output filename prefix')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.report_file).exists():
        print(f"❌ Error: Report file not found: {args.report_file}")
        sys.exit(1)
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║                       PyBench Table Generator                                   ║
    ║                   Convert JSON Reports to Comprehensive Tables                  ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Initialize generator
        generator = PyBenchTableGenerator(args.report_file)
        
        # Generate tables
        generated_files = generator.generate_all_tables(
            formats=args.format,
            output_prefix=args.output
        )
        
        print(f"\n✅ Table generation complete!")
        
        # Show generated files
        if generated_files:
            print(f"\n📁 Generated files:")
            for format_type, files in generated_files.items():
                if isinstance(files, list):
                    print(f"   {format_type.upper()}: {len(files)} files")
                    for file in files[:3]:  # Show first 3
                        print(f"     • {Path(file).name}")
                    if len(files) > 3:
                        print(f"     • ... and {len(files)-3} more")
                else:
                    print(f"   {format_type.upper()}: {Path(files).name}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()