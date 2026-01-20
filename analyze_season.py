"""
Analyze season evaluation results to identify improvement opportunities.
"""
import pandas as pd
import numpy as np

# Load data
summary = pd.read_csv('reports/season_2025/2025_season_summary.csv')
misplacements = pd.read_csv('reports/season_2025/2025_season_misplacements.csv')

print("=" * 80)
print("SEASON 2025 ANALYSIS")
print("=" * 80)

# Overall stats
print(f"\n📊 OVERALL PERFORMANCE:")
print(f"   Average Spearman: {summary['Correlation'].mean():.3f}")
print(f"   Median Spearman: {summary['Correlation'].median():.3f}")
print(f"   Std Dev: {summary['Correlation'].std():.3f}")
print(f"   Min: {summary['Correlation'].min():.3f} ({summary.loc[summary['Correlation'].idxmin(), 'Race']})")
print(f"   Max: {summary['Correlation'].max():.3f} ({summary.loc[summary['Correlation'].idxmax(), 'Race']})")

# Categorize races
excellent = summary[summary['Correlation'] >= 0.85]
good = summary[(summary['Correlation'] >= 0.7) & (summary['Correlation'] < 0.85)]
mediocre = summary[(summary['Correlation'] >= 0.5) & (summary['Correlation'] < 0.7)]
poor = summary[summary['Correlation'] < 0.5]

print(f"\n📈 PERFORMANCE DISTRIBUTION:")
print(f"   Excellent (≥0.85): {len(excellent)} races")
print(f"   Good (0.7-0.85):   {len(good)} races")
print(f"   Mediocre (0.5-0.7): {len(mediocre)} races")
print(f"   Poor (<0.5):        {len(poor)} races")

print(f"\n❌ POOREST PERFORMING RACES:")
worst = summary.nsmallest(5, 'Correlation')[['Race', 'Correlation', 'PodiumHits']]
for idx, row in worst.iterrows():
    print(f"   {row['Race']:30s} Corr: {row['Correlation']:6.3f}  Podium: {row['PodiumHits']}/3")

print(f"\n✅ BEST PERFORMING RACES:")
best = summary.nlargest(5, 'Correlation')[['Race', 'Correlation', 'PodiumHits']]
for idx, row in best.iterrows():
    print(f"   {row['Race']:30s} Corr: {row['Correlation']:6.3f}  Podium: {row['PodiumHits']}/3")

# Misplacement analysis
print(f"\n📏 MISPLACEMENT ANALYSIS:")
print(f"   Average Absolute Misplacement: {misplacements['AbsMisplacement'].mean():.2f} positions")
print(f"   Median Absolute Misplacement: {misplacements['AbsMisplacement'].median():.2f} positions")
print(f"   Max Misplacement: {misplacements['AbsMisplacement'].max():.0f} positions")

# DNF/DNQ impact (drivers finishing position 20)
dnf_drivers = misplacements[misplacements['Actual'] == 20]
print(f"\n🚫 DNF/DQ IMPACT (Actual finish = 20):")
print(f"   Total DNF cases: {len(dnf_drivers)}")
print(f"   Average predicted position for DNFs: {dnf_drivers['Predicted'].mean():.1f}")
print(f"   Average misplacement for DNFs: {dnf_drivers['AbsMisplacement'].mean():.1f} positions")

# Most mispredicted drivers
print(f"\n👤 MOST MISPREDICTED DRIVERS:")
driver_mispl = misplacements.groupby('Driver')['AbsMisplacement'].agg(['mean', 'count', 'max'])
driver_mispl = driver_mispl[driver_mispl['count'] >= 10].sort_values('mean', ascending=False).head(10)
for driver, row in driver_mispl.iterrows():
    print(f"   {driver}: Avg {row['mean']:.2f} (races: {row['count']:.0f}, max: {row['max']:.0f})")

# Worst individual predictions
print(f"\n🎯 WORST INDIVIDUAL PREDICTIONS:")
worst_pred = misplacements.nlargest(10, 'AbsMisplacement')[['Race', 'Driver', 'Start', 'Actual', 'Predicted', 'Misplacement']]
for idx, row in worst_pred.iterrows():
    print(f"   {row['Race']:25s} {row['Driver']:3s} Start:{row['Start']:2.0f} → Actual:{row['Actual']:2.0f} Pred:{row['Predicted']:2.0f} (Off by {row['Misplacement']:3.0f})")

# Pattern analysis: Starting position vs prediction accuracy
print(f"\n🏁 GRID POSITION ACCURACY:")
misplacements['GridGroup'] = pd.cut(misplacements['Start'], bins=[0, 5, 10, 20], labels=['Top 5', 'Mid 6-10', 'Back 11+'])
grid_acc = misplacements.groupby('GridGroup')['AbsMisplacement'].mean()
for grid, mispl in grid_acc.items():
    print(f"   {grid:10s}: {mispl:.2f} avg misplacement")

# DNF prediction failure
print(f"\n🔍 KEY ISSUES IDENTIFIED:")
print("   1. DNF/Reliability: Model doesn't predict retirements (avg {:.1f} positions off)".format(dnf_drivers['AbsMisplacement'].mean()))
print("   2. Race incidents: Unexpected crashes/penalties heavily skew predictions")
print("   3. Strategy surprises: Late-race strategy calls not captured")

# Poor race deep dive
print(f"\n🔬 POOR RACES DEEP DIVE:")
for race_name in poor['Race'].tolist():
    race_data = misplacements[misplacements['Race'] == race_name]
    corr = summary[summary['Race'] == race_name]['Correlation'].values[0]
    dnfs = len(race_data[race_data['Actual'] == 20])
    avg_mispl = race_data['AbsMisplacement'].mean()
    print(f"\n   {race_name} (Corr: {corr:.3f}):")
    print(f"      DNFs/DQs: {dnfs}")
    print(f"      Avg misplacement: {avg_mispl:.2f}")
    print(f"      Top 3 worst predictions:")
    for idx, row in race_data.nlargest(3, 'AbsMisplacement').iterrows():
        print(f"         {row['Driver']} P{row['Start']:.0f}→{row['Actual']:.0f} pred:{row['Predicted']:.0f} (off {row['Misplacement']:+.0f})")

print("\n" + "=" * 80)
print("IMPROVEMENT RECOMMENDATIONS")
print("=" * 80)
print("""
1. ⚠️ DNF/Reliability Modeling:
   - Add historical reliability factors per driver/team
   - Implement probabilistic DNF prediction based on track history
   - Weight predictions by reliability confidence

2. 🏎️ Race Incident Handling:
   - Track per-track incident rates (safety cars, crashes)
   - Model first-lap chaos factor (especially street circuits)
   - Add uncertainty bands for high-incident tracks

3. 🎯 Strategy Variation:
   - Expand strategy space beyond 1-2 stop rigid plans
   - Model undercut/overcut dynamics more explicitly
   - Consider alternative compound strategies (Med-Hard, etc.)

4. 🧮 Feature Improvements:
   - Weight recent form more heavily for driver-track combinations
   - Add tyre management skill differentiation (currently generic)
   - Model qualifying vs race pace gap per driver

5. 📊 Track-Specific Tuning:
   - Street circuits (Monaco, Singapore, Vegas): higher consistency, lower overtaking
   - High-speed tracks (Monza, Spa): tire deg more critical
   - Hot tracks (Qatar, Bahrain): reliability/cooling factors

6. 🔢 Simulation Parameters:
   - Increase sim count for high-variance tracks (street circuits)
   - Tune consistency sigma per track type
   - Add late-race fatigue/pressure factors
""")
