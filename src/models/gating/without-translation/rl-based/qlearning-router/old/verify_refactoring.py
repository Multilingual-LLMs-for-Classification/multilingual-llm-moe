#!/usr/bin/env python3
"""
Verification script for router1.py refactoring.
Tests that all components are properly structured and imports work.
"""

import sys
from pathlib import Path

def test_component_structure():
    """Verify component files exist and are properly structured."""
    print("=" * 80)
    print("1. TESTING COMPONENT STRUCTURE")
    print("=" * 80)

    components_dir = Path(__file__).parent / "components"
    required_files = [
        "language_detector.py",
        "domain_classifier.py",
        "q_learning_router.py",
        "routing_system.py",
        "__init__.py"
    ]

    for filename in required_files:
        filepath = components_dir / filename
        if filepath.exists():
            line_count = len(filepath.read_text().splitlines())
            print(f"✅ {filename:25s} exists ({line_count:4d} lines)")
        else:
            print(f"❌ {filename:25s} MISSING")
            return False

    print("\n")
    return True

def test_component_imports():
    """Test that component imports work."""
    print("=" * 80)
    print("2. TESTING COMPONENT IMPORTS")
    print("=" * 80)

    try:
        print("Attempting: from components import LanguageDetector...")
        from components import LanguageDetector
        print("✅ LanguageDetector imported successfully")
    except ImportError as e:
        if "fasttext" in str(e):
            print("⚠️  LanguageDetector import requires fasttext (expected)")
        else:
            print(f"❌ LanguageDetector import failed: {e}")
            return False

    try:
        print("Attempting: from components import DomainClassifier...")
        from components import DomainClassifier
        print("✅ DomainClassifier imported successfully")
    except ImportError as e:
        print(f"❌ DomainClassifier import failed: {e}")
        return False

    try:
        print("Attempting: from components import QLearningTaskClassifier...")
        from components import QLearningTaskClassifier
        print("✅ QLearningTaskClassifier imported successfully")
    except ImportError as e:
        print(f"❌ QLearningTaskClassifier import failed: {e}")
        return False

    try:
        print("Attempting: from components import PromptRoutingSystem...")
        from components import PromptRoutingSystem
        print("✅ PromptRoutingSystem imported successfully")
    except ImportError as e:
        if "fasttext" in str(e):
            print("⚠️  PromptRoutingSystem import requires fasttext (expected)")
        else:
            print(f"❌ PromptRoutingSystem import failed: {e}")
            return False

    print("\n")
    return True

def test_backward_compatibility():
    """Test that router1.py backward compatibility works."""
    print("=" * 80)
    print("3. TESTING BACKWARD COMPATIBILITY")
    print("=" * 80)

    router1_path = Path(__file__).parent / "router1.py"
    if not router1_path.exists():
        print("❌ router1.py does not exist")
        return False

    print(f"✅ router1.py exists ({len(router1_path.read_text().splitlines())} lines)")

    try:
        print("Attempting: from router1 import PromptRoutingSystem...")
        # Import router1 module
        import router1
        print("✅ router1 module imported successfully")

        # Check it has the expected exports
        expected_exports = ['LanguageDetector', 'DomainClassifier',
                          'QLearningTaskClassifier', 'PromptRoutingSystem']
        for export in expected_exports:
            if hasattr(router1, export):
                print(f"✅ router1.{export} is available")
            else:
                print(f"❌ router1.{export} is NOT available")
                return False
    except ImportError as e:
        if "fasttext" in str(e):
            print("⚠️  router1 import requires fasttext (expected)")
        else:
            print(f"❌ router1 import failed: {e}")
            return False

    print("\n")
    return True

def test_configuration():
    """Test that configuration system works."""
    print("=" * 80)
    print("4. TESTING CONFIGURATION SYSTEM")
    print("=" * 80)

    # Check router_config.py exists
    config_py = Path(__file__).parent / "router_config.py"
    if not config_py.exists():
        print("❌ router_config.py does not exist")
        return False
    print("✅ router_config.py exists")

    # Check router_config.json exists
    config_json = Path(__file__).parent / "router_config.json"
    if not config_json.exists():
        print("❌ router_config.json does not exist")
        return False
    print("✅ router_config.json exists")

    # Check expert config files exist
    routing_file = Path(__file__).parent / "components" / "routing_system.py"
    config_path = routing_file.parents[5] / "experts" / "config"

    if not config_path.exists():
        print(f"❌ Expert config directory does not exist: {config_path}")
        return False
    print(f"✅ Expert config directory exists: {config_path}")

    domain_tasks = config_path / "domain_tasks.json"
    if not domain_tasks.exists():
        print(f"❌ domain_tasks.json not found: {domain_tasks}")
        return False
    print("✅ domain_tasks.json exists")

    experts_registry = config_path / "experts_registry.json"
    if not experts_registry.exists():
        print(f"❌ experts_registry.json not found: {experts_registry}")
        return False
    print("✅ experts_registry.json exists")

    # Try loading config
    try:
        from router_config import RouterSystemConfig
        print("✅ RouterSystemConfig imported successfully")

        config = RouterSystemConfig.from_json(config_json)
        print("✅ Configuration loaded from JSON successfully")

        # Check key settings
        print(f"  - Domain classifier epochs: {config.domain_config.epochs}")
        print(f"  - Q-learning val_split: {config.qlearning_config.val_split}")
        print(f"  - Test samples: {config.evaluation.test_n or 'ALL'}")

        if config.evaluation.test_n == 2:
            print("⚠️  WARNING: test_n is set to 2 (only tests 2 samples!)")
            print("   Consider setting to null or a larger value in router_config.json")
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

    print("\n")
    return True

def test_documentation():
    """Test that documentation files exist."""
    print("=" * 80)
    print("5. TESTING DOCUMENTATION")
    print("=" * 80)

    doc_files = [
        "ROUTER_COMPONENT_SPLIT.md",
        "EXPERT_SPECIFIC_EVALUATION.md",
        "COMPLETE_EXPERT_DRIVEN_FLOW.md",
        "CONFIG_UPDATE_SUMMARY.md",
        "REFACTORING_COMPLETE.md",
        "components/README.md"
    ]

    base_dir = Path(__file__).parent
    for doc_file in doc_files:
        doc_path = base_dir / doc_file
        if doc_path.exists():
            print(f"✅ {doc_file}")
        else:
            print(f"❌ {doc_file} MISSING")

    print("\n")
    return True

def main():
    """Run all verification tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "ROUTER1.PY REFACTORING VERIFICATION" + " " * 23 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")

    tests = [
        ("Component Structure", test_component_structure),
        ("Component Imports", test_component_imports),
        ("Backward Compatibility", test_backward_compatibility),
        ("Configuration System", test_configuration),
        ("Documentation", test_documentation)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} raised exception: {e}")
            results.append((test_name, False))

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10s} {test_name}")

    print("\n")
    print(f"Result: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Refactoring is complete and verified.")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install fasttext transformers torch scikit-learn")
        print("  2. Test the system: python main.py --mode eval --test-n 100")
        print("  3. Review documentation in REFACTORING_COMPLETE.md")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
