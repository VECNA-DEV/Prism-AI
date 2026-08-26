"""Quick syntax and import validation for Prism AI modules."""
import py_compile
import glob
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

files = sorted(glob.glob("prism/**/*.py", recursive=True))
files += sorted(glob.glob("scripts/*.py"))

print(f"Checking {len(files)} Python files...")

errors = []
for f in files:
    try:
        py_compile.compile(f, doraise=True)
    except py_compile.PyCompileError as e:
        errors.append((f, str(e)))
        print(f"  FAIL: {f}")
        print(f"        {e}")

if errors:
    print(f"\n{len(errors)} files have syntax errors!")
else:
    print(f"\nAll {len(files)} files compile successfully!")

# Test config creation
try:
    from prism.model.config import PrismConfig
    config = PrismConfig()
    config.validate()
    print(f"\nPrismConfig created: {config.num_params_billions:.2f}B parameters")
    print(f"  hidden_size={config.hidden_size}, layers={config.num_layers}")
    print(f"  heads={config.num_attention_heads}q/{config.num_kv_heads}kv")
    print(f"  head_dim={config.head_dim}, intermediate={config.intermediate_size}")
except Exception as e:
    print(f"\nConfig test failed: {e}")
