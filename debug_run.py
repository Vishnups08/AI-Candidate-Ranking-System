import traceback
import sys
from rank import main

try:
    sys.argv = ['rank.py', '--candidates', r'c:\Users\Vishnu\Music\New folder\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\sample_candidates.json', '--out', 'output/submission_sample.xlsx', '--skip-embeddings']
    main()
except Exception:
    traceback.print_exc()
