"""
Quick test script for monitoring integration
Run this to verify your hybrid monitoring is working
"""
import requests
import json
import sys
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"
COLORS = {
    'GREEN': '\033[92m',
    'RED': '\033[91m',
    'YELLOW': '\033[93m',
    'BLUE': '\033[94m',
    'END': '\033[0m'
}

def print_color(text, color='GREEN'):
    """Print colored text"""
    print(f"{COLORS.get(color, '')}{text}{COLORS['END']}")

def check_api_running():
    """Check if API is accessible"""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=2)
        return response.status_code == 200
    except:
        return False

def test_endpoint(name, method, endpoint, data=None):
    """Test a single endpoint"""
    print(f"\n{'='*60}")
    print_color(f"Testing: {name}", 'BLUE')
    print(f"{'='*60}")
    
    try:
        if method == 'GET':
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
        else:
            response = requests.post(f"{BASE_URL}{endpoint}", json=data, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2))
            print_color(f"✅ {name} - PASSED", 'GREEN')
            return True, result
        else:
            print_color(f"❌ {name} - FAILED (Status: {response.status_code})", 'RED')
            print(response.text)
            return False, None
            
    except Exception as e:
        print_color(f"❌ {name} - ERROR: {e}", 'RED')
        return False, None

def check_monitoring_files():
    """Check if monitoring files exist"""
    print(f"\n{'='*60}")
    print_color("Checking Monitoring Files", 'BLUE')
    print(f"{'='*60}")
    
    files = {
        'monitor.py': 'monitoring/monitor.py',
        'metrics.json': 'monitoring/metrics.json',
        'baseline.json': 'monitoring/baseline.json'
    }
    
    for name, path in files.items():
        exists = Path(path).exists()
        if exists:
            print_color(f"✅ {name}: Found", 'GREEN')
        else:
            status = 'YELLOW' if name in ['metrics.json', 'baseline.json'] else 'RED'
            msg = "Not created yet (normal on first run)" if name in ['metrics.json', 'baseline.json'] else "MISSING!"
            print_color(f"⚠️  {name}: {msg}", status)

def main():
    """Run all tests"""
    print_color("\n" + "="*60, 'BLUE')
    print_color("MONITORING INTEGRATION TEST SUITE", 'BLUE')
    print_color("="*60, 'BLUE')
    
    # Check if API is running
    print("\nChecking if API is running...")
    if not check_api_running():
        print_color("\n❌ API is not running!", 'RED')
        print_color("Please start the API first:", 'YELLOW')
        print_color("  cd model && python api/main.py", 'YELLOW')
        sys.exit(1)
    
    print_color("✅ API is running\n", 'GREEN')
    
    # Check monitoring files
    check_monitoring_files()
    
    # Run tests
    tests = [
        {
            'name': 'Health Check',
            'method': 'GET',
            'endpoint': '/api/health'
        },
        {
            'name': 'Monitor Status',
            'method': 'GET',
            'endpoint': '/api/monitor/status'
        },
        {
            'name': 'Monitor Health',
            'method': 'GET',
            'endpoint': '/api/monitor/health'
        },
        {
            'name': 'Retraining Check',
            'method': 'GET',
            'endpoint': '/api/monitor/retraining-check'
        },
        {
            'name': 'Query Test',
            'method': 'POST',
            'endpoint': '/api/query',
            'data': {
                'query': 'What are the latest AI trends?',
                'filters': None,
                'enable_streaming': False
            }
        }
    ]
    
    results = []
    for test in tests:
        passed, result = test_endpoint(
            test['name'],
            test['method'],
            test['endpoint'],
            test.get('data')
        )
        results.append((test['name'], passed, result))
        time.sleep(1)
    
    # Print summary
    print(f"\n{'='*60}")
    print_color("TEST SUMMARY", 'BLUE')
    print(f"{'='*60}\n")
    
    passed_count = sum(1 for _, passed, _ in results if passed)
    total_count = len(results)
    
    for name, passed, _ in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        color = 'GREEN' if passed else 'RED'
        print_color(f"{status} - {name}", color)
    
    print(f"\n{'-'*60}")
    print_color(f"Results: {passed_count}/{total_count} tests passed", 
                'GREEN' if passed_count == total_count else 'RED')
    print(f"{'-'*60}")
    
    # Check monitoring mode
    for name, passed, result in results:
        if name == 'Monitor Status' and passed:
            mode = result.get('mode', 'unknown')
            print(f"\n📊 Monitoring Mode: {mode}")
            
            if result.get('simple_monitor', {}).get('available'):
                print_color("  ✅ Simple Monitor: Active", 'GREEN')
            
            if result.get('gcp_monitor', {}).get('available'):
                print_color("  ✅ GCP Monitor: Active", 'GREEN')
            elif mode == 'local_only':
                print_color("  ℹ️  GCP Monitor: Not enabled (using local only)", 'YELLOW')
                print_color("     To enable GCP: export GCP_PROJECT_ID=your-project-id", 'YELLOW')
    
    # Check if metrics were logged
    print(f"\n{'='*60}")
    print_color("POST-TEST CHECKS", 'BLUE')
    print(f"{'='*60}\n")
    
    metrics_file = Path('monitoring/metrics.json')
    if metrics_file.exists():
        print_color("✅ Metrics file created", 'GREEN')
        try:
            with open(metrics_file) as f:
                metrics = json.load(f)
                print(f"   Total queries logged: {len(metrics)}")
                if metrics:
                    latest = metrics[-1]
                    print(f"   Latest query:")
                    print(f"     - Validation: {latest.get('validation_score', 0):.3f}")
                    print(f"     - Fairness: {latest.get('fairness_score', 0):.3f}")
                    print(f"     - Response time: {latest.get('response_time', 0):.2f}s")
        except Exception as e:
            print_color(f"   ⚠️  Could not read metrics: {e}", 'YELLOW')
    else:
        print_color("⚠️  Metrics file not yet created", 'YELLOW')
        print("   This is normal on first run - send a query to create it")
    
    # Final verdict
    print(f"\n{'='*60}")
    if passed_count == total_count:
        print_color("✅ ALL TESTS PASSED - MONITORING IS WORKING!", 'GREEN')
    elif passed_count > 0:
        print_color(f"⚠️  PARTIAL SUCCESS - {total_count - passed_count} test(s) failed", 'YELLOW')
    else:
        print_color("❌ ALL TESTS FAILED - CHECK CONFIGURATION", 'RED')
    print(f"{'='*60}\n")
    
    # Next steps
    if passed_count == total_count:
        print_color("🎯 Next Steps:", 'BLUE')
        print("1. Run 50+ queries to build baseline")
        print("2. Establish baseline: curl -X POST http://localhost:8000/api/monitor/establish-baseline")
        print("3. Monitor health regularly: curl http://localhost:8000/api/monitor/health")
        print("4. (Optional) Enable GCP monitoring for production features\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_color("\n\n⚠️  Test interrupted by user", 'YELLOW')
        sys.exit(0)
    except Exception as e:
        print_color(f"\n\n❌ Unexpected error: {e}", 'RED')
        import traceback
        traceback.print_exc()
        sys.exit(1)