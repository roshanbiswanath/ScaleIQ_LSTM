import re

# Fix the model comparison file
with open('data_analysis/model_comparison_analysis.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the timestamp column reference  
content = content.replace("pd.to_datetime(self.data['timestamp'])", "pd.to_datetime(self.data['DateTime'])")

# Add queue_length mapping right after timestamp
content = content.replace(
    "self.data['timestamp'] = pd.to_datetime(self.data['DateTime'])",
    "self.data['timestamp'] = pd.to_datetime(self.data['DateTime'])\n        self.data['queue_length'] = self.data['avg_unprocessed_events_count']"
)

with open('data_analysis/model_comparison_analysis.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ Fixed model comparison file column references')
