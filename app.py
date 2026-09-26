evaluation_rows = []

test_results = [test_1, test_2, test_3, test_4, test_5]

for i, (_, gt) in enumerate(ground_truth.iterrows()):
    tr = test_results[i]

    evaluation_rows.append({
        'Test Case': gt['Test Case'],
        'Expected Route': gt['Expected Route'],
        'Actual Route': tr['route'],
        'Route Match': tr['route'] == gt['Expected Route'],
        'Expected Query ID': gt['Expected Query ID'],
        'Actual Query ID': tr['query_id'],
        'Query ID Match': (
            pd.isna(gt['Expected Query ID']) and pd.isna(tr['query_id'])
        ) or tr['query_id'] == gt['Expected Query ID'],
        'Confidence': tr['confidence'],
        'Rows Returned': tr['row_count']
    })

evaluation_df = pd.DataFrame(evaluation_rows)

path_accuracy = evaluation_df['Route Match'].mean() * 100

verified = evaluation_df['Expected Route'].str.strip().str.lower() == 'verified'
query_accuracy = evaluation_df.loc[verified, 'Query ID Match'].mean() * 100

# Coerce to numeric so a string like 'ESCALATED' becomes NaN instead of crashing .mean()
numeric_confidence = pd.to_numeric(evaluation_df['Confidence'], errors='coerce')
average_confidence = numeric_confidence.mean()

print(f"Selected Path Accuracy: {path_accuracy:.1f}%")
print(f"Selected Query Accuracy: {query_accuracy:.1f}%")
print(f"Average Confidence Score: {average_confidence:.2f}")

if evaluation_df['Confidence'].astype(str).eq('ESCALATED').any():
    n_escalated = evaluation_df['Confidence'].astype(str).eq('ESCALATED').sum()
    print(f"Note: {n_escalated} test case(s) were escalated to a human analyst and excluded from the confidence average.")

evaluation_df
