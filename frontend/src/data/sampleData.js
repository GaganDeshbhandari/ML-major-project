export const navigation = [
  { label: 'Dashboard', iconName: 'dashboard' },
  { label: 'Claims', iconName: 'claims' },
  { label: 'Model evaluation', iconName: 'evaluation' },
];

export const objectives = [
  { id: 'false-claims', apiValue: 'remove_false_claims', title: 'Remove False Claims', description: 'Prioritize claims most likely to be false.' },
  { id: 'misconceptions', apiValue: 'correct_user_misconceptions', title: 'Correct User-Misconceptions', description: 'Prioritize claims that users perceive incorrectly.' },
  { id: 'disagreement', apiValue: 'reduce_user_disagreement', title: 'Reduce User Disagreement', description: 'Prioritize claims with the most disagreement.' },
];

export function getObjectiveTitle(objectiveId) {
  return (
    objectives.find((item) => item.id === objectiveId || item.apiValue === objectiveId)?.title
    || 'Remove False Claims'
  );
}

export function getObjectiveApiValue(objectiveId) {
  return (
    objectives.find((item) => item.id === objectiveId || item.apiValue === objectiveId)?.apiValue
    || 'remove_false_claims'
  );
}
