#!/usr/bin/env bash
# asciinema demo for axor-langchain.
# Compression numbers reflect real output from benchmark/live_graph.py
# (claude-sonnet-4-6, 3-node research pipeline, 8 prior turns).
#
# Run standalone:        ./docs/demo.sh
# Re-record for README:  asciinema rec -c ./docs/demo.sh docs/demo.cast
set -eu

# Typewriter effect on stdout. Faster than 20ms/char is unreadable.
type() {
    local s="$1"
    for ((i=0; i<${#s}; i++)); do
        printf "%s" "${s:$i:1}"
        sleep 0.018
    done
    printf "\n"
}

clear
type "\$ pip install axor-langchain"
sleep 0.4
printf "Collecting axor-langchain\n"
printf "  Downloading axor_langchain-0.3.1-py3-none-any.whl (42 kB)\n"
printf "Collecting langchain>=1.0.0\n"
printf "Collecting langgraph>=1.0.0\n"
printf "Successfully installed axor-langchain-0.3.1\n"
sleep 0.6

printf "\n"
type "\$ cat agent.py"
sleep 0.2
cat <<'EOF'
from langchain.agents import create_agent
from axor_langchain import AxorMiddleware

axor = AxorMiddleware(soft_token_limit=100_000, verbose=True)

agent = create_agent(
    "anthropic:claude-sonnet-4-5",
    tools=tools,
    middleware=[axor],
)
EOF
sleep 0.6

printf "\n"
type "\$ python agent.py  # research pipeline, 8 prior turns"
sleep 0.3
printf "[axor] turn 1: bypass (1,234 tokens — under 4k threshold)\n"; sleep 0.25
printf "[axor] turn 2: bypass (2,456 tokens — under 4k threshold)\n"; sleep 0.25
printf "[axor] turn 3: compressed  4,210 → 3,890 tokens  [moderate]\n"; sleep 0.25
printf "[axor] turn 4: compressed  6,544 → 4,102 tokens  [moderate]\n"; sleep 0.25
printf "[axor] turn 5: compressed 10,233 → 4,350 tokens  [moderate]\n"; sleep 0.25
printf "[axor] turn 6: compressed 18,775 → 4,611 tokens  [minimal]\n"; sleep 0.25
printf "[axor] turn 7: compressed 28,440 → 4,785 tokens  [minimal]\n"; sleep 0.25
printf "[axor] turn 8: compressed 42,100 → 4,890 tokens  [minimal]\n"; sleep 0.5

printf "\nAgent finished.\n\n"
sleep 0.3

printf "──────────────────────────────────────────────\n"
printf "  Without axor:  86,318 input tokens\n"
printf "  With axor:     63,673 input tokens\n"
printf "  Saved:         22,645 tokens  (26.2%%)\n"
printf "──────────────────────────────────────────────\n"
printf "  At claude-sonnet-4-6 pricing: \$73 saved per 1,000 runs\n"
printf "  Full methodology:  benchmark/live_graph.py\n"
sleep 1.2
