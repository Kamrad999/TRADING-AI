#!/usr/bin/env python3
"""
Long-run stability test for TRADING-AI production validation.

Simulates 6 hours of continuous RSS polling to detect memory growth,
cache bloat, connection leaks, and system stability issues.
"""

import sys
import time
import psutil
import os
import threading
from pathlib import Path
from datetime import datetime, timezone
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_ai.core.orchestrator import PipelineOrchestrator
from trading_ai.infrastructure.config import config


class StabilityMonitor:
    """Monitor system stability during long-run test."""
    
    def __init__(self):
        self.start_time = time.time()
        self.initial_memory = psutil.Process().memory_info().rss
        self.initial_disk = psutil.disk_usage('.').used
        self.process = psutil.Process()
        self.metrics = {
            'memory_samples': [],
            'cpu_samples': [],
            'disk_samples': [],
            'connection_counts': [],
            'error_counts': [],
            'restart_times': []
        }
    
    def record_metrics(self):
        """Record current system metrics."""
        memory_info = psutil.Process().memory_info().rss
        cpu_percent = psutil.cpu_percent()
        disk_usage = psutil.disk_usage('.')
        
        self.metrics['memory_samples'].append(memory_info)
        self.metrics['cpu_samples'].append(cpu_percent)
        self.metrics['disk_samples'].append(disk_usage.used)
        
        # Calculate memory growth
        memory_growth = memory_info - self.initial_memory
        disk_growth = disk_usage.used - self.initial_disk
        
        return {
            'elapsed_time': time.time() - self.start_time,
            'memory_current': memory_info,
            'memory_growth': memory_growth,
            'memory_growth_mb': memory_growth / (1024 * 1024),
            'cpu_percent': cpu_percent,
            'disk_used': disk_usage.used,
            'disk_growth_mb': disk_growth / (1024 * 1024)
        }
    
    def check_health(self):
        """Check overall system health."""
        try:
            orchestrator = PipelineOrchestrator()
            status = orchestrator.get_system_status()
            
            return {
                'healthy': True,
                'system_status': status,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }


def run_rss_polling_test(duration_hours=6):
    """Run continuous RSS polling for specified duration."""
    print(f"Starting {duration_hours}-hour RSS polling stability test...")
    print(f"Start time: {datetime.now(timezone.utc)}")
    
    monitor = StabilityMonitor()
    orchestrator = PipelineOrchestrator()
    
    start_time = time.time()
    end_time = start_time + (duration_hours * 3600)
    
    poll_count = 0
    error_count = 0
    
    while time.time() < end_time:
        try:
            print(f"\n[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Poll #{poll_count + 1}")
            
            # Run one pipeline cycle
            result = orchestrator.run_pipeline(dry_run=True)
            
            if result.success:
                poll_count += 1
                print(f"✅ Pipeline completed successfully")
            else:
                error_count += 1
                print(f"❌ Pipeline failed: {result.error_message}")
            
            # Record metrics
            metrics = monitor.record_metrics()
            
            print(f"📊 Memory: {metrics['memory_growth_mb']:.1f}MB growth")
            print(f"💾 Disk: {metrics['disk_growth_mb']:.1f}MB growth")
            print(f"⚡ CPU: {metrics['cpu_percent']:.1f}%")
            
            # Check for concerning patterns
            if metrics['memory_growth_mb'] > 100:  # 100MB memory growth
                print(f"⚠️ WARNING: High memory growth detected")
            
            if error_count > 5:
                print(f"🚨 CRITICAL: High error rate: {error_count}")
            
            # Wait for next poll (5 minutes)
            time.sleep(300)
            
        except KeyboardInterrupt:
            print("\n🛑 Test interrupted by user")
            break
        except Exception as e:
            print(f"\n💥 Unexpected error: {e}")
            error_count += 1
            time.sleep(60)  # Wait 1 minute on error
    
    # Final health check
    print(f"\n{'='*60}")
    print("FINAL STABILITY CHECK")
    health = monitor.check_health()
    
    if health['healthy']:
        print("✅ System healthy after long-run test")
    else:
        print(f"❌ System unhealthy: {health['error']}")
    
    # Calculate final statistics
    total_time = time.time() - start_time
    avg_memory_growth = sum(m['memory_growth_mb'] for m in monitor.metrics['memory_samples']) / len(monitor.metrics['memory_samples'])
    
    print(f"\n📈 FINAL STATISTICS:")
    print(f"Total runtime: {total_time/3600:.1f} hours")
    print(f"Total polls: {poll_count}")
    print(f"Total errors: {error_count}")
    print(f"Average memory growth: {avg_memory_growth:.1f}MB")
    print(f"Error rate: {error_count/max(poll_count, 1)*100:.1f}%")
    
    # Check state file size
    state_file = Path(config.STATE_FILE)
    if state_file.exists():
        state_size = state_file.stat().st_size / (1024 * 1024)  # MB
        print(f"State file size: {state_size:.1f}MB")
        
        if state_size > 10:  # 10MB state file
            print("⚠️ WARNING: Large state file detected")
    
    return {
        'total_runtime_hours': total_time / 3600,
        'total_polls': poll_count,
        'total_errors': error_count,
        'error_rate_percent': error_count/max(poll_count, 1)*100,
        'avg_memory_growth_mb': avg_memory_growth,
        'final_health': health['healthy'],
        'state_file_size_mb': state_size if state_file.exists() else 0
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Long-run stability test for TRADING-AI")
    parser.add_argument("--duration", type=int, default=6, 
                       help="Test duration in hours (default: 6)")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick 30-minute test")
    
    args = parser.parse_args()
    
    duration = args.duration if not args.quick else 0.5  # 30 minutes for quick test
    
    results = run_rss_polling_test(duration)
    
    # Determine verdict
    critical_issues = []
    
    if results['error_rate_percent'] > 5:
        critical_issues.append("High error rate")
    
    if results['avg_memory_growth_mb'] > 50:
        critical_issues.append("Excessive memory growth")
    
    if results['state_file_size_mb'] > 5:
        critical_issues.append("Large state file")
    
    if not results['final_health']:
        critical_issues.append("System unhealthy")
    
    print(f"\n🚀 STABILITY VERDICT:")
    if critical_issues:
        print("❌ CRITICAL ISSUES FOUND:")
        for issue in critical_issues:
            print(f"  - {issue}")
        print("\n🔧 FIXES REQUIRED BEFORE PAPER TRADING")
        sys.exit(1)
    else:
        print("✅ SYSTEM STABLE FOR PAPER TRADING")
        print(f"Error rate: {results['error_rate_percent']:.1f}%")
        print(f"Memory growth: {results['avg_memory_growth_mb']:.1f}MB")
        sys.exit(0)
