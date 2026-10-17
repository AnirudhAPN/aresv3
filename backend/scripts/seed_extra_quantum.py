import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Quantum Machine Learning Approaches for Combinatorial Optimization"

PLANNER_JSON = {
    "research_objectives": [
        "Investigate how variational quantum algorithms, particularly the Quantum Approximate Optimization Algorithm (QAOA) and quantum-enhanced reinforcement learning, can be applied to solve NP-hard combinatorial optimization problems such as Max-Cut, graph coloring, and vehicle routing.",
        "Characterize the trade-offs between solution quality, circuit depth, and qubit count for quantum machine learning models compared to classical heuristics (simulated annealing, genetic algorithms, tabu search) on equivalent problem instances.",
        "Design hybrid quantum-classical training pipelines that mitigate barren plateaus and noise-induced degradation when optimizing parameterized quantum circuits for combinatorial cost functions.",
        "Assess the practical scalability of near-term quantum machine learning approaches to combinatorial optimization on current NISQ (noisy intermediate-scale quantum) hardware versus quantum simulators.",
    ],
    "research_questions": [
        "How does the performance of QAOA-based and quantum neural network solvers scale with problem size and circuit depth on both simulated and real NISQ devices?",
        "What hybrid quantum-classical optimization strategies (e.g., parameter-shift gradients combined with classical optimizers like COBYLA or Adam) most effectively reduce training instability caused by barren plateaus?",
        "To what extent do quantum-enhanced solvers outperform state-of-the-art classical metaheuristics on standard combinatorial benchmarks under realistic noise models?",
        "What problem encodings (QUBO formulations, graph embeddings) most efficiently map real-world combinatorial optimization tasks onto available qubit topologies?",
    ],
    "keywords": [
        "quantum approximate optimization algorithm",
        "variational quantum circuits",
        "combinatorial optimization",
        "NISQ quantum computing",
        "quantum-classical hybrid learning",
    ],
    "research_scope": (
        "This research focuses on quantum machine learning methods applied specifically to combinatorial optimization "
        "problems, including QAOA, variational quantum eigensolvers adapted for optimization, and quantum-enhanced "
        "reinforcement learning for solving problems such as Max-Cut, traveling salesman, graph coloring, and "
        "QUBO-formulated scheduling tasks. It covers hybrid quantum-classical training loops, noise mitigation "
        "strategies, and benchmarking against classical heuristics on both quantum simulators and NISQ hardware. "
        "It excludes quantum algorithms for unstructured search or factoring (e.g., Grover's or Shor's algorithm) "
        "that are not framed as combinatorial optimization, and does not cover fault-tolerant quantum computing "
        "architectures assuming error-corrected qubits beyond the current NISQ era."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Widespread use of QAOA with shallow circuit depths (p=1 to p=3) as the default variational ansatz for Max-Cut and QUBO-style combinatorial benchmarks.",
        "Growing interest in warm-start and problem-informed circuit initialization strategies to accelerate convergence and reduce the impact of barren plateaus.",
        "Increasing reliance on quantum simulators (e.g., statevector and noisy density-matrix simulators) rather than physical NISQ hardware for reporting headline results, due to hardware qubit and coherence-time limits.",
        "Emerging hybrid approaches that pair classical graph neural networks with quantum circuits to precompute problem embeddings before variational optimization.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic comparison of quantum machine learning solvers against strong classical metaheuristics (e.g., tuned simulated annealing) on identically sized, real-world combinatorial instances rather than small synthetic graphs.",
            "based_on_papers": ["A Quantum Approximate Optimization Algorithm"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: standardized noise-mitigation protocols that would let results reported on different NISQ hardware backends be directly compared for combinatorial optimization tasks.",
            "based_on_papers": ["Barren Plateaus in Quantum Neural Network Training Landscapes"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: rigorous characterization of how QUBO problem encoding choices affect the trainability and final solution quality of variational quantum optimizers, independent of hardware noise.",
            "based_on_papers": ["Quantum Machine Learning"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Development of open benchmark suites pairing real-world combinatorial optimization instances with matched classical and quantum baselines run under comparable compute budgets.",
        "Research into adaptive-depth or problem-aware ansatz design to jointly address barren plateaus and hardware noise without sacrificing expressivity.",
        "Systematic studies isolating the effect of QUBO encoding strategy from hardware noise effects on solution quality for variational quantum optimization.",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {
            "name": "OpenML",
            "url": "https://www.openml.org/",
            "description": "Searchable repository of ML datasets with standardized benchmarking metadata.",
            "verified": True,
        }
    ],
    "algorithms": [
        {
            "name": "Quantum Approximate Optimization Algorithm (QAOA)",
            "url": "https://arxiv.org/abs/1411.4028",
            "description": "Variational quantum-classical hybrid algorithm that alternates problem and mixer Hamiltonians to approximate solutions to combinatorial optimization problems encoded as Ising or QUBO cost functions.",
            "verified": True,
        },
        {
            "name": "Reinforcement Learning (PPO)",
            "url": "https://arxiv.org/abs/1707.06347",
            "description": "Proximal Policy Optimization used as a classical outer-loop controller for adaptively tuning variational quantum circuit parameters and ansatz structure during hybrid quantum-classical training.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Approximation ratio relative to optimal or best-known classical solution",
        "Circuit depth and two-qubit gate count",
        "Convergence rate (iterations to reach target approximation ratio)",
        "Solution quality degradation under realistic hardware noise models",
    ],
    "implementation_steps": [
        "Formulate the target combinatorial optimization problem (e.g., Max-Cut, graph coloring) as a QUBO or Ising Hamiltonian suitable for variational quantum encoding.",
        "Select and initialize a parameterized quantum circuit ansatz (QAOA layers or hardware-efficient ansatz) matched to the available qubit connectivity.",
        "Run the hybrid optimization loop, using a classical optimizer to update circuit parameters based on expectation values estimated via the quantum device or simulator.",
        "Apply noise mitigation techniques (zero-noise extrapolation, readout error correction) when executing on NISQ hardware backends.",
        "Benchmark the resulting approximate solutions against classical heuristics (simulated annealing, tabu search) on matched problem instances and compute budgets.",
        "Analyze scaling behavior by repeating the pipeline across increasing problem sizes and circuit depths to characterize the accuracy-resource trade-off.",
    ],
    "unverified_suggestions": ["IBM Quantum Experience benchmark suite", "D-Wave QUBO benchmark set", "MQLib combinatorial optimization library"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal grounds the research in a well-established variational algorithm (QAOA) with a clear, standard problem encoding (QUBO/Ising), making the approach reproducible and comparable to prior work.",
        "The methodology explicitly incorporates noise mitigation strategies, reflecting realistic awareness of NISQ hardware limitations rather than assuming idealized quantum execution.",
        "Pairing a classical reinforcement learning controller with the quantum circuit optimization loop is a reasonable and increasingly common strategy for addressing trainability challenges like barren plateaus.",
        "The implementation plan is logically sequenced, moving from problem encoding through circuit design, hybrid training, noise mitigation, and classical baseline comparison.",
    ],
    "weaknesses": [
        "The evidence base is limited to a small set of retrieved papers, so conclusions about gaps such as encoding-versus-noise effects should be treated as preliminary rather than definitive.",
        "OpenML, the only verified dataset source, is not well suited to combinatorial optimization benchmarking, which typically relies on graph instance libraries or QUBO benchmark sets rather than tabular ML datasets, so dataset fit needs further validation.",
        "The proposal does not fully specify qubit count and connectivity assumptions, which strongly affect which problem sizes are even feasible to test on current hardware.",
        "There is limited discussion of statistical variance across repeated quantum circuit executions (shot noise), which could confound comparisons of approximation ratio between quantum and classical methods.",
    ],
    "missing_components": [
        "A concrete protocol for selecting classical baseline hyperparameters (e.g., annealing schedule) to ensure fair, compute-matched comparison against the quantum solver.",
        "An ablation plan isolating the contribution of the RL-based parameter controller versus standard gradient-based classical optimizers for QAOA training.",
    ],
    "novelty_score": 0.72,
    "quality_score": 0.76,
}

CONFIDENCE_JSON = {"internal_consistency": 0.82, "specificity": 0.79, "completeness": 0.84}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to quantum machine learning for combinatorial optimization, based on the retrieved abstract." for t in titles]
        return json.dumps({"summaries": summaries})
    if "Gap Analysis Agent" in prompt:
        return json.dumps(GAP_JSON)
    if "Methodology Agent" in prompt:
        return json.dumps(METHODOLOGY_JSON)
    raise AssertionError(f"Unexpected generate() prompt: {prompt[:200]}")


def mock_judge(prompt, temperature=0.0, agent=None, **kwargs):
    if "internal_consistency" in prompt:
        return json.dumps(CONFIDENCE_JSON)
    if "qualitative critique" in prompt or "Reviewer Agent" in prompt:
        return json.dumps(REVIEWER_JSON)
    raise AssertionError(f"Unexpected judge() prompt: {prompt[:200]}")


def main():
    from services.memory import memory_service
    from services.llm import llm_service
    import orchestrator as orch

    session_id = memory_service.create_session(TOPIC, source="seed")
    print("session_id:", session_id)

    with patch.object(llm_service, "generate", side_effect=mock_generate), \
         patch.object(llm_service, "judge", side_effect=mock_judge):
        final_state = orch.run_pipeline(session_id, TOPIC, condition="adaptive")

    print("completed_agents:", final_state.get("completed_agents"))
    print("acs_scores:", json.dumps(final_state.get("acs_scores", {}), indent=2))
    print("flags:", final_state.get("flags"))
    session = memory_service.get_session(session_id)
    print("FINAL STATUS:", session["status"])


if __name__ == "__main__":
    main()
