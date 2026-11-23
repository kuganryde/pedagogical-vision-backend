/**
 * Node.js Connection Example for Pedagogical Vision Backend API
 * Compatible with Node.js 18+ (uses native fetch)
 */

const API_BASE = 'http://localhost:8088/api/v1';

async function main() {
  console.log('Connecting Node.js module to Pedagogical Vision Backend...');

  // 1. Start classroom session
  const sessionRes = await fetch(`${API_BASE}/session/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: 'node_classroom_session_1',
      classroom_name: 'CS Lab 3',
      course_name: 'Operating Systems',
      sliding_window_seconds: 10.0
    })
  });
  const sessionData = await sessionRes.json();
  console.log('Session started:', sessionData);

  // 2. Send frame for analysis
  const frameRes = await fetch(`${API_BASE}/session/node_classroom_session_1/frame`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      frame_id: 1,
      timestamp: Date.now() / 1000
    })
  });
  const frameData = await frameRes.json();
  console.log('Classroom Engagement Score:', frameData.mean_engagement_score);
  console.log('Students Detected:', frameData.total_students_detected);
  console.log('Pedagogical Alerts:', frameData.active_pedagogical_alerts);

  // 3. Close session and retrieve pedagogical report
  const stopRes = await fetch(`${API_BASE}/session/node_classroom_session_1/stop`, {
    method: 'POST'
  });
  const summary = await stopRes.json();
  console.log('Lecture Summary Report:', summary);
}

main().catch(err => console.error('Error:', err));
