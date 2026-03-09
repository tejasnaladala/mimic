import React from 'react'

interface EpisodeManagerProps {
  isConnected: boolean
  recording: boolean
  episodeCount: number
  sendCommand: (cmd: Record<string, unknown>) => void
}

export default function EpisodeManager({
  isConnected,
  recording,
  episodeCount,
  sendCommand,
}: EpisodeManagerProps) {
  return (
    <div className="panel">
      <div className="panel__title">Episodes</div>
      <div className="episode-controls">
        <div className="episode-counter">
          {episodeCount}
          <div className="episode-counter__label">Episodes Collected</div>
        </div>

        <div className="btn-group">
          <button
            className="btn btn--primary btn--sm"
            onClick={() => sendCommand({ type: 'start_recording' })}
            disabled={!isConnected || recording}
          >
            Start Recording
          </button>
          <button
            className="btn btn--ghost btn--sm"
            onClick={() => sendCommand({ type: 'stop_recording' })}
            disabled={!isConnected || !recording}
          >
            Stop
          </button>
        </div>

        <div className="btn-group">
          <button
            className="btn btn--success btn--sm"
            onClick={() => sendCommand({ type: 'save_episode' })}
            disabled={!isConnected || recording}
          >
            Save Episode
          </button>
          <button
            className="btn btn--warning btn--sm"
            onClick={() => sendCommand({ type: 'discard_episode' })}
            disabled={!isConnected || recording}
          >
            Discard
          </button>
        </div>
      </div>
    </div>
  )
}
