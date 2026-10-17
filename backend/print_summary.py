import csv, json, sys

# Load metrics
with open('evaluation_results.csv', encoding='utf-8') as f:
    metrics = list(csv.DictReader(f))

# Load sessions
with open('evaluation_sessions.json', encoding='utf-8') as f:
    sessions = json.load(f)

CONDITIONS = ['single_agent', 'static', 'adaptive']
AGENTS = ['planner', 'literature', 'gap', 'methodology', 'reviewer']

print('ARES EVALUATION RESULTS')
print('=' * 80)

# Success rate
print('\n--- Run Success Rate ---')
for s in sessions:
    status = s.get('status', 'unknown')

# Count by condition
from collections import Counter
condition_status = Counter()
for s in sessions:
    # Determine condition from acs_scores
    has_acs = bool(s.get('acs_scores'))
    print(f"  {s['topic']}: {s['status']}")

print(f'\nTotal sessions: {len(sessions)}')
completed = sum(1 for s in sessions if s['status'] == 'completed')
failed = sum(1 for s in sessions if s['status'] == 'failed')
print(f'Completed: {completed}, Failed: {failed}')

# Average ACS by condition and agent
print('\n--- Average ACS by Condition x Agent ---')
header = f"{'Agent':<15s}"
for cond in CONDITIONS:
    header += f"  {cond:>12s}"
print(header)
print('-' * 55)

for agent in AGENTS:
    row = f"{agent:<15s}"
    for cond in CONDITIONS:
        scores = [float(m['acs']) for m in metrics if m['agent_name'] == agent and m['condition'] == cond and m['acs']]
        avg = sum(scores) / len(scores) if scores else 0
        row += f"  {avg:>12.3f}"
    print(row)

# Single agent baseline
sa_scores = [float(m['acs']) for m in metrics if m['condition'] == 'single_agent' and m['acs']]
print(f"\n{'baseline':<15s}  {(sum(sa_scores)/len(sa_scores) if sa_scores else 0):>12.3f}")

# ACS component breakdown for adaptive
print('\n--- ACS Component Breakdown (Adaptive) ---')
print(f"{'Agent':<15s}  {'Confidence':>12s}  {'Evidence':>12s}  {'Completion':>12s}  {'ACS':>12s}")
print('-' * 67)
for agent in AGENTS:
    am = [m for m in metrics if m['agent_name'] == agent and m['condition'] == 'adaptive']
    if not am:
        continue
    avg_c = sum(float(m['confidence'] or 0) for m in am) / len(am)
    avg_e = sum(float(m['evidence_quality'] or 0) for m in am) / len(am)
    avg_t = sum(float(m['task_completion'] or 0) for m in am) / len(am)
    avg_acs = sum(float(m['acs'] or 0) for m in am) / len(am)
    print(f"{agent:<15s}  {avg_c:>12.3f}  {avg_e:>12.3f}  {avg_t:>12.3f}  {avg_acs:>12.3f}")

# Execution time
print('\n--- Avg Execution Time per Agent (Adaptive, seconds) ---')
for agent in AGENTS:
    times = [float(m['execution_time']) for m in metrics if m['agent_name'] == agent and m['condition'] == 'adaptive' and m['execution_time']]
    avg_t = sum(times) / len(times) if times else 0
    print(f"  {agent:<15s}: {avg_t:>8.1f}s")

# Coherence
print('\n--- Cross-Agent Coherence (Adaptive sessions) ---')
coh_scores = [s['coherence']['overall_score'] for s in sessions if s.get('coherence')]
gap_cov = [s['coherence']['gap_coverage_score'] for s in sessions if s.get('coherence')]
q_align = [s['coherence']['question_alignment_score'] for s in sessions if s.get('coherence')]
if coh_scores:
    print(f"  Overall coherence:     mean={sum(coh_scores)/len(coh_scores):.3f}  min={min(coh_scores):.3f}  max={max(coh_scores):.3f}")
    print(f"  Gap coverage:          mean={sum(gap_cov)/len(gap_cov):.3f}")
    print(f"  Question alignment:    mean={sum(q_align)/len(q_align):.3f}")

# Retries and failures
print('\n--- Retry and Parse Failure Summary ---')
total_retries = sum(sum(s.get('retry_counts', {}).values()) for s in sessions)
total_parse_fails = sum(sum(1 for v in s.get('parse_failures', {}).values() if v) for s in sessions)
print(f"  Total ACS retries: {total_retries}")
print(f"  Total parse failures: {total_parse_fails}")

# Flags
print('\n--- Flags Raised ---')
all_flags = {}
for s in sessions:
    for flag in s.get('flags', []):
        all_flags[flag] = all_flags.get(flag, 0) + 1
for flag, count in sorted(all_flags.items(), key=lambda x: -x[1]):
    print(f"  {flag}: {count}")

print('\n' + '=' * 80)
