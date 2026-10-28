export const navigation = [
  { label: 'Dashboard', iconName: 'dashboard' },
  { label: 'Prioritization runs', iconName: 'runs' },
  { label: 'Claims', iconName: 'claims' },
  { label: 'Model evaluation', iconName: 'evaluation' },
  { label: 'Settings', iconName: 'settings' },
];

export const objectives = [
  { id: 'false-claims', title: 'Remove False Claims', description: 'Prioritize claims most likely to be false.' },
  { id: 'misconceptions', title: 'Correct User-Misconceptions', description: 'Prioritize claims that users perceive incorrectly.' },
  { id: 'disagreement', title: 'Reduce User Disagreement', description: 'Prioritize claims with the most disagreement.' },
];

export const sampleClaims = [
  { id: 'CLM-0248', claim: "The city's new policy reduced traffic congestion by 40%.", baseScore: 0.91, ecs: 0.82, priority: 'High', status: 'Mixed evidence', source: 'City transport report and local traffic data show different baselines.' },
  { id: 'CLM-0187', claim: 'A new learning app improves exam scores by 25%.', baseScore: 0.86, ecs: 0.74, priority: 'High', status: 'Contradicting', source: 'The available studies report a smaller improvement across different cohorts.' },
  { id: 'CLM-0203', claim: 'Urban air pollution decreased across the district.', baseScore: 0.79, ecs: 0.39, priority: 'Medium', status: 'Supporting', source: 'Several monitoring stations support a downward trend in particulate matter.' },
  { id: 'CLM-0114', claim: 'Daily screen time has no effect on sleep quality.', baseScore: 0.76, ecs: 0.91, priority: 'High', status: 'Conflicting', source: 'Sleep studies point to different effects depending on age and usage patterns.' },
  { id: 'CLM-0221', claim: 'Public transit use increased after fare changes.', baseScore: 0.71, ecs: 0.32, priority: 'Medium', status: 'Supporting', source: 'Transit authority counts show an increase during the first reporting period.' },
  { id: 'CLM-0069', claim: 'The coastal cleanup removed most plastic waste from the shoreline.', baseScore: 0.64, ecs: 0.67, priority: 'Medium', status: 'Mixed evidence', source: 'Cleanup totals are high, but the measured shoreline coverage is incomplete.' },
  { id: 'CLM-0155', claim: 'Remote work leads to higher productivity for every team.', baseScore: 0.58, ecs: 0.88, priority: 'High', status: 'Contradicting', source: 'Productivity results vary substantially by role, team structure, and measurement.' },
  { id: 'CLM-0092', claim: 'The community garden program increased local food access.', baseScore: 0.52, ecs: 0.28, priority: 'Low', status: 'Supporting', source: 'Program participation and distribution records support the claim.' },
  { id: 'CLM-0236', claim: 'Recycling reminders doubled household participation.', baseScore: 0.48, ecs: 0.61, priority: 'Medium', status: 'Conflicting', source: 'Participation rose, although the available sample is smaller than the claim suggests.' },
];

export function getObjectiveTitle(objectiveId) {
  return objectives.find((item) => item.id === objectiveId)?.title || 'Remove False Claims';
}
