const test = require('node:test');
const assert = require('node:assert/strict');

const { state, projectStep, isCaptionExport, isActiveProject, shouldReconnect, previousRenderPayload, friendlyProjectError, actionableWarnings, nearestReplacement, nextProgressValue } = require('../static/app.js');

test('a review project never inherits the export step', () => {
  const reviewProject = { status: 'review', analysis: { segments: [{ text: 'Line' }] } };
  const completeProject = { status: 'complete', analysis: { segments: [{ text: 'Line' }] } };
  const qualityReviewProject = { status: 'quality_review', analysis: { segments: [{ text: 'Line' }] } };

  assert.equal(projectStep(reviewProject), 'transcript');
  assert.equal(projectStep(completeProject), 'result');
  assert.equal(projectStep(qualityReviewProject), 'result');
});

test('updates from a previous project or navigation are rejected', () => {
  state.navigation = 8;
  state.project = { id: 'current-project' };

  assert.equal(isActiveProject('current-project', 8), true);
  assert.equal(isActiveProject('previous-project', 8), false);
  assert.equal(isActiveProject('current-project', 7), false);
});

test('a dropped project socket reconnects only for the active project', () => {
  const socket = {};
  state.navigation = 9;
  state.project = { id: 'active-project' };
  state.socket = socket;

  assert.equal(shouldReconnect('active-project', 9, socket), true);
  assert.equal(shouldReconnect('old-project', 9, socket), false);
  assert.equal(shouldReconnect('active-project', 8, socket), false);
  assert.equal(shouldReconnect('active-project', 9, {}), false);
});

test('retry reuses only the supported render settings', () => {
  const payload = previousRenderPayload({ render: { target_language: 'zh', voice_mode: 'clone', run_id: 'old', unexpected: true } });
  assert.deepEqual(payload, {
    target_language: 'zh', voice_mode: 'clone', background_volume: 1, expression: 0.5, quality: 'high',
  });
});

test('retry preserves the lip sync choice', () => {
  const payload = previousRenderPayload({ render: { target_language: 'en', voice_mode: 'clone', lip_sync_enabled: true } });
  assert.equal(payload.lip_sync_enabled, true);
});

test('structured translation failures are explained without a traceback', () => {
  const message = friendlyProjectError('Traceback\njson.decoder.JSONDecodeError: Extra data');
  assert.match(message, /retry the dub/i);
  assert.doesNotMatch(message, /Traceback/);
});

test('voice similarity is informational while other QA notices remain visible', () => {
  const warnings = actionableWarnings({ warnings: [
    'Low voice similarity for SPEAKER_00: 0.71',
    'Integrated loudness outside target range: -18 LUFS',
  ] });

  assert.deepEqual(warnings, ['Integrated loudness outside target range: -18 LUFS']);
});

test('caption-only re-export stays on the export step', () => {
  const project = { status: 'rendering', render: { caption_only: true }, analysis: { segments: [{ text: 'Line' }] } };
  assert.equal(isCaptionExport(project), true);
  assert.equal(projectStep(project), 'result');
});

test('deleted speaker lines move to the nearest remaining speaker', () => {
  const segments = [
    { start: 0, end: 2, speaker: 'A' },
    { start: 3, end: 4, speaker: 'WRONG' },
    { start: 8, end: 10, speaker: 'B' },
  ];

  assert.equal(nearestReplacement(segments, 'WRONG', 1, 'A'), 'A');
});

test('visible progress advances smoothly one percent at a time', () => {
  assert.equal(nextProgressValue(30, 56), 31);
  assert.equal(nextProgressValue(55, 56), 56);
  assert.equal(nextProgressValue(80, 30), 30);
});
