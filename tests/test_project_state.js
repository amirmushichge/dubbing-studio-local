const test = require('node:test');
const assert = require('node:assert/strict');

const { state, projectStep, isActiveProject } = require('../static/app.js');

test('a review project never inherits the export step', () => {
  const reviewProject = { status: 'review', analysis: { segments: [{ text: 'Line' }] } };
  const completeProject = { status: 'complete', analysis: { segments: [{ text: 'Line' }] } };

  assert.equal(projectStep(reviewProject), 'transcript');
  assert.equal(projectStep(completeProject), 'result');
});

test('updates from a previous project or navigation are rejected', () => {
  state.navigation = 8;
  state.project = { id: 'current-project' };

  assert.equal(isActiveProject('current-project', 8), true);
  assert.equal(isActiveProject('previous-project', 8), false);
  assert.equal(isActiveProject('current-project', 7), false);
});
