#This is the code file for the quiz in Complexity Lab

#xn+1 = 3xn -xn^3

import numpy as np
import matplotlib.pyplot as plt

# Define the map function
def f(x):
    return 3*x - x**3

# Define the derivative of the map
def f_prime(x):
    return 3 - 3*x**2

print("Analysis of the map: xn+1 = 3xn - xn^3")
print("="*50)

# Find fixed points: x* = f(x*) => x* = 3x* - (x*)^3
# Rearranging: 0 = 3x* - (x*)^3 - x* = 2x* - (x*)^3 = x*(2 - (x*)^2)
# So either x* = 0 or (x*)^2 = 2

print("\n1. Finding Fixed Points:")
print("Setting x* = f(x*): x* = 3x* - (x*)^3")
print("Rearranging: 0 = 2x* - (x*)^3 = x*(2 - (x*)^2)")
print("Solutions: x* = 0 or (x*)^2 = 2")

fixed_points = [0, np.sqrt(2), -np.sqrt(2)]
print(f"\nFixed points: x* = 0, x* = √2 ≈ {np.sqrt(2):.4f}, x* = -√2 ≈ {-np.sqrt(2):.4f}")

print("\n2. Stability Analysis:")
print("For stability, we check |f'(x*)| where f'(x) = 3 - 3x^2")
print("If |f'(x*)| < 1: stable")
print("If |f'(x*)| > 1: unstable")
print("If |f'(x*)| = 1: marginal (needs further analysis)")

for i, x_star in enumerate(fixed_points):
    derivative = f_prime(x_star)
    stability = "stable" if abs(derivative) < 1 else "unstable" if abs(derivative) > 1 else "marginal"
    
    if x_star == 0:
        print(f"\nAt x* = 0:")
    elif x_star > 0:
        print(f"\nAt x* = √2 ≈ {x_star:.4f}:")
    else:
        print(f"\nAt x* = -√2 ≈ {x_star:.4f}:")
    
    print(f"  f'(x*) = 3 - 3({x_star:.4f})^2 = {derivative:.4f}")
    print(f"  |f'(x*)| = {abs(derivative):.4f}")
    print(f"  Status: {stability}")

print("\n" + "="*50)
print("SUMMARY:")
print("All three fixed points (0, √2, -√2) are UNSTABLE")
print("since |f'(x*)| > 1 for all fixed points.")




import matplotlib.pyplot as plt
import numpy as np

# Create x values for plotting
x = np.linspace(-2.5, 2.5, 1000)

# Plot the function f(x) = 3x - x^3
plt.figure(figsize=(10, 6))
plt.plot(x, f(x), 'b-', label='f(x) = 3x - x³')
plt.plot(x, x, 'r--', label='y = x')

# Plot fixed points
for x_star in fixed_points:
    plt.plot(x_star, f(x_star), 'ko')

# Add labels and title
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Graph of f(x) = 3x - x³ and Fixed Points')
plt.grid(True)
plt.legend()

# # Show the plot
# plt.show()








#xn+1 = aplpha xn

print("\n\n" + "="*60)
print("Analysis of Linear Map: xn+1 = α·xn")
print("="*60)

# Define the linear map
def linear_map(x, alpha):
    return alpha * x

def linear_map_derivative(alpha):
    return alpha

print("\n1. Lyapunov Exponent Calculation:")
print("For the map xn+1 = α·xn, f'(x) = α (constant)")
print("The Lyapunov exponent is: λ = ln|α|")

# Test different values of alpha
alpha_values = [0.5, 0.8, 1.0, 1.2, 2.0, -0.5, -0.8, -1.0, -1.2, -2.0]

print(f"\n{'α':<8} {'|α|':<8} {'λ = ln|α|':<12} {'Behavior'}")
print("-" * 50)

for alpha in alpha_values:
    abs_alpha = abs(alpha)
    if abs_alpha == 0:
        lambda_exp = float('-inf')
        behavior = "Fixed at 0"
    elif abs_alpha == 1:
        lambda_exp = 0.0
        behavior = "Marginal"
    else:
        lambda_exp = np.log(abs_alpha)
        if abs_alpha < 1:
            behavior = "Stable"
        else:
            behavior = "Unstable"
    
    print(f"{alpha:<8.1f} {abs_alpha:<8.1f} {lambda_exp:<12.4f} {behavior}")

print("\n2. Theoretical Analysis:")
print("For xn+1 = α·xn:")
print("• f(x) = α·x")
print("• f'(x) = α (constant)")
print("• Lyapunov exponent: λ = ln|α|")

print("\n3. Interpretation:")
print("• λ < 0 (|α| < 1): Stable - orbits converge to 0")
print("• λ = 0 (|α| = 1): Marginal - orbits stay constant or oscillate")
print("• λ > 0 (|α| > 1): Unstable - orbits diverge exponentially")

print("\n4. Special Cases:")
print("• α = 0: All orbits immediately go to 0 (super-stable)")
print("• α = 1: All orbits stay constant (neutral)")  
print("• α = -1: Period-2 oscillation between x and -x")
print("• |α| > 1: Exponential growth/decay depending on sign")

# Demonstrate with time series for different alpha values
print("\n5. Time Series Examples:")
x0 = 1.0  # Initial condition
n_steps = 10

test_alphas = [0.5, 1.0, 1.5, -0.5, -1.0, -1.5]

for alpha in test_alphas:
    print(f"\nα = {alpha}, x0 = {x0}:")
    x = x0
    trajectory = [x]
    for i in range(n_steps):
        x = linear_map(x, alpha)
        trajectory.append(x)
    
    # Show first few and last few values
    print(f"  Trajectory: {trajectory[:4]} ... {trajectory[-3:]}")
    print(f"  λ = ln|{alpha}| = {np.log(abs(alpha)):.4f}")

print("\n" + "="*60)
print("SUMMARY FOR LINEAR MAP:")
print("The Lyapunov exponent λ = ln|α| completely determines the dynamics:")
print("• Stable (λ < 0): |α| < 1")
print("• Marginal (λ = 0): |α| = 1") 
print("• Unstable (λ > 0): |α| > 1")
print("="*60)


# Create a figure with two subplots
plt.figure(figsize=(12, 5))

# Plot 1: Cobweb plot
plt.subplot(121)
x = np.linspace(-2, 2, 1000)
for alpha in [0.5, 1.0, 1.5]:
    plt.plot(x, linear_map(x, alpha), label=f'α = {alpha}')
plt.plot(x, x, 'k--', label='y = x')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Linear Map: f(x) = αx')
plt.legend()

# Plot 2: Time series
plt.subplot(122)
x0 = 1.0
n_steps = 20
for alpha in [0.5, 1.0, 1.5]:
    x = x0
    trajectory = [x]
    for i in range(n_steps):
        x = linear_map(x, alpha)
        trajectory.append(x)
    plt.plot(range(len(trajectory)), trajectory, 'o-', label=f'α = {alpha}')

plt.grid(True)
plt.xlabel('n')
plt.ylabel('x')
plt.title('Time Series')
plt.legend()

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("EXPLAINING 'ALL VALUES OF ALPHA'")
print("="*60)

print("\nWhen we say 'find the Lyapunov exponent for ALL values of α',")
print("we mean find a GENERAL FORMULA that works for ANY real number α.")
print("\nFor the linear map xn+1 = α·xn:")

print("\n1. ANALYTICAL SOLUTION (works for all α):")
print("   λ = ln|α|")
print("   This single formula covers the ENTIRE parameter space!")

print("\n2. PARAMETER SPACE ANALYSIS:")
print("   Let α ∈ ℝ (α can be any real number)")
print("   Then the Lyapunov exponent is:")
print("   ")
print("   ⎧ -∞         if α = 0")
print("   ⎪ ln|α| < 0  if 0 < |α| < 1  (stable)")
print("   λ = ⎨ 0           if |α| = 1      (marginal)")
print("   ⎪ ln|α| > 0  if |α| > 1      (unstable)")
print("   ⎩")

print("\n3. WHY THIS IS 'ALL VALUES':")
print("   • For ANY positive α: λ = ln(α)")
print("   • For ANY negative α: λ = ln(-α) = ln|α|") 
print("   • This covers the entire real line: α ∈ (-∞, +∞)")

print("\n4. VERIFICATION WITH SPECIFIC VALUES:")
test_values = [-100, -2, -1, -0.5, 0, 0.5, 1, 2, 100]
print("   α        |α|       λ = ln|α|")
print("   " + "-"*30)
for alpha in test_values:
    if alpha == 0:
        print(f"   {alpha:<8} {abs(alpha):<8} -∞")
    else:
        print(f"   {alpha:<8} {abs(alpha):<8.1f} {np.log(abs(alpha)):<8.4f}")

print("\n5. KEY INSIGHT:")
print("   Unlike nonlinear maps where we might need numerical methods,")
print("   the LINEAR map has a CLOSED-FORM solution for λ.")
print("   The formula λ = ln|α| is EXACT for every possible α ≠ 0.")

print("\n6. MATHEMATICAL DERIVATION:")
print("   For xn+1 = f(xn) = α·xn:")
print("   • f'(x) = α (constant derivative)")
print("   • λ = lim(n→∞) (1/n) Σ ln|f'(xi)|")
print("   • λ = lim(n→∞) (1/n) Σ ln|α|")
print("   • λ = lim(n→∞) (1/n) · n · ln|α|")
print("   • λ = ln|α|")

print("\n" + "="*60)
print("ANSWER: The Lyapunov exponent for ALL values of α is:")
print("λ = ln|α|")
print("This single formula completely characterizes the dynamics")
print("for the entire infinite parameter space α ∈ ℝ.")
print("="*60)


import numpy as np
import matplotlib.pyplot as plt

# Set parameters
alpha = 100
n_iterations = 100
x0 = 0.1  # Initial condition

# Generate sequence
x = np.zeros(n_iterations)
x[0] = x0
for i in range(1, n_iterations):
    x[i] = alpha * x[i-1]

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(range(n_iterations), x, 'b-', label=f'α = {alpha}')
plt.xlabel('Iteration (n)')
plt.ylabel('xₙ')
plt.title('Linear Map: xₙ₊₁ = α·xₙ')
plt.grid(True)
plt.legend()
plt.yscale('log')  # Use log scale since values grow exponentially
plt.show()

print("\n" + "="*60)
print("WHY THE LINEAR MAP IS NOT (AND CANNOT BE) CHAOTIC")
print("="*60)

print("\nYou're absolutely right! The linear map xn+1 = α·xn is:")
print("• SUPER SIMPLE")
print("• COMPLETELY PREDICTABLE") 
print("• NEVER CHAOTIC")

print("\n1. WHAT CHAOS REQUIRES:")
print("   For a system to be chaotic, it needs:")
print("   ✓ Sensitive dependence on initial conditions")
print("   ✓ Topological mixing")
print("   ✓ Dense periodic orbits")
print("   ✓ NONLINEARITY (this is crucial!)")

print("\n2. WHY LINEAR MAPS CAN'T BE CHAOTIC:")

print("\n   a) TOO PREDICTABLE:")
print("      • xn = α^n · x0 (exact solution!)")
print("      • You can predict ANY future state exactly")
print("      • No 'butterfly effect' - small changes stay small (if |α| < 1)")

print("\n   b) LIMITED DYNAMICS:")
print("      • Only 4 possible behaviors:")
print("        - Converge to 0 (|α| < 1)")
print("        - Stay constant (α = 1)")
print("        - Flip between ±x0 (α = -1)") 
print("        - Grow/shrink exponentially (|α| > 1)")

print("\n   c) NO STRANGE ATTRACTORS:")
print("      • Attractors are just points (0) or infinity")
print("      • No complex geometric structures")

print("\n3. DEMONSTRATION - PREDICTABILITY:")
print("   Let's show how predictable it is:")

# Show exact prediction vs chaos
alphas_demo = [0.9, 1.1, -0.9, -1.1]
x0 = 0.5

for alpha in alphas_demo:
    print(f"\n   α = {alpha}, x0 = {x0}:")
    
    # Exact formula
    x10_exact = alpha**10 * x0
    x20_exact = alpha**20 * x0
    
    # Iterative calculation
    x = x0
    for i in range(10):
        x = alpha * x
    x10_iter = x
    
    for i in range(10):
        x = alpha * x
    x20_iter = x
    
    print(f"   x10: Exact = {x10_exact:.6f}, Computed = {x10_iter:.6f}")
    print(f"   x20: Exact = {x20_exact:.6f}, Computed = {x20_iter:.6f}")
    print(f"   → Perfect agreement! No chaos here.")

print("\n4. WHAT YOU NEED FOR CHAOS:")
print("   Examples of chaotic maps:")
print("   • Logistic map: xn+1 = r·xn(1-xn)")
print("   • Tent map: xn+1 = r·min(xn, 1-xn)")  
print("   • Hénon map: xn+1 = 1 - ax²n + yn")
print("   → All have NONLINEAR terms!")

print("\n5. THE KEY INSIGHT:")
print("   NONLINEARITY IS ESSENTIAL FOR CHAOS")
print("   • Linear → predictable, boring")
print("   • Nonlinear → can be chaotic, interesting")

print("\n6. LYAPUNOV EXPONENT PERSPECTIVE:")
print("   • Chaos requires λ > 0 AND complex dynamics")
print("   • Linear map: λ = ln|α|")
print("   • Even when λ > 0 (|α| > 1), it's just exponential growth")
print("   • No sensitivity to initial conditions in the chaotic sense")

print("\n" + "="*60)
print("CONCLUSION:")
print("The linear map is a 'toy model' to understand Lyapunov exponents,")
print("but it's TOO SIMPLE to be chaotic. Real chaos needs nonlinearity!")
print("Think of it as 'training wheels' before studying truly chaotic systems.")
print("="*60)
