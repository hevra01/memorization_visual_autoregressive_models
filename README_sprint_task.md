# The VAR repository "https://github.com/FoundationVision/VAR" is extended to complete the task.

# Check the memorization folder, which has the code related to the memorization task.

# The only change made to the base VAR repo is in models/var.py, __init__ where the attention mask is changed from attending all the previous scales (s-1, s-2, etc) to only attending to the immediate previous scale (s-1)