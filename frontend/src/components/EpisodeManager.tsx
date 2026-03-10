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
    <div className="hud-episode">
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        {recording && (
          <div className="hud-rec">
            <span className="hud-rec__dot" />
            REC
          </div>
        )}
        <div>
          <div className="hud-episode__counter">{episodeCount}</div>
          <div className="hud-episode__label">EPISODES</div>
        </div>
      </div>

      <div className="hud-btn-group">
        {!recording ? (
          <button
            className="hud-btn hud-btn--danger"
            onClick={() => sendCommand({ type: 'start_recording' })}
            disabled={!isConnected}
          >
            [ REC ]
          </button>
        ) : (
          <button
            className="hud-btn"
            onClick={() => sendCommand({ type: 'stop_recording' })}
            disabled={!isConnected}
          >
            [ STOP ]
          </button>
        )}
        <button
          className="hud-btn"
          onClick={() => sendCommand({ type: 'save_episode' })}
          disabled={!isConnected || recording}
        >
          [ SAVE ]
        </button>
        <button
          className="hud-btn hud-btn--danger"
          onClick={() => sendCommand({ type: 'discard_episode' })}
          disabled={!isConnected || recording}
        >
          [ DISCARD ]
        </button>
      </div>
    </div>
  )
}
