import React from 'react'

interface ControlsProps {
  isConnected: boolean
  onConnect: () => void
  onDisconnect: () => void
  sendCommand: (cmd: Record<string, unknown>) => void
  envInfo: EnvInfo | null
  controlMode: string
  onModeChange: (mode: string) => void
  stats: Stats
}

interface EnvInfo {
  name: string
  action_dim: number
  state_dim: number
  control_hz: number
  cameras: Array<{ name: string; width: number; height: number }>
}

interface Stats {
  reward: number
  done: boolean
  isSuccess: boolean
  recording: boolean
}

export default function Controls({
  isConnected,
  onConnect,
  onDisconnect,
  sendCommand,
  envInfo,
  controlMode,
  onModeChange,
  stats,
}: ControlsProps) {
  const handleModeChange = (mode: string) => {
    onModeChange(mode)
    sendCommand({ type: 'set_mode', mode })
  }

  return (
    <div className="sidebar">
      {/* Connection */}
      <div className="panel">
        <div className="panel__title">Connection</div>
        <div className="btn-group">
          {!isConnected ? (
            <button className="btn btn--primary" onClick={onConnect}>
              Connect
            </button>
          ) : (
            <button className="btn btn--danger" onClick={onDisconnect}>
              Disconnect
            </button>
          )}
          <button
            className="btn btn--ghost"
            onClick={() => sendCommand({ type: 'reset' })}
            disabled={!isConnected}
          >
            Reset Env
          </button>
        </div>
      </div>

      {/* Environment Info */}
      {envInfo && (
        <div className="panel">
          <div className="panel__title">Environment</div>
          <div className="info-row">
            <span className="info-row__label">Name</span>
            <span className="info-row__value">{envInfo.name}</span>
          </div>
          <div className="info-row">
            <span className="info-row__label">Action Dim</span>
            <span className="info-row__value">{envInfo.action_dim}</span>
          </div>
          <div className="info-row">
            <span className="info-row__label">Control Hz</span>
            <span className="info-row__value">{envInfo.control_hz}</span>
          </div>
          <div className="info-row">
            <span className="info-row__label">Cameras</span>
            <span className="info-row__value">
              {envInfo.cameras.map((c) => c.name).join(', ')}
            </span>
          </div>
        </div>
      )}

      {/* Control Mode */}
      <div className="panel">
        <div className="panel__title">Control Mode</div>
        <div className="mode-toggle">
          <button
            className={`mode-toggle__option ${
              controlMode === 'joint' ? 'mode-toggle__option--active' : ''
            }`}
            onClick={() => handleModeChange('joint')}
          >
            Joint
          </button>
          <button
            className={`mode-toggle__option ${
              controlMode === 'cartesian' ? 'mode-toggle__option--active' : ''
            }`}
            onClick={() => handleModeChange('cartesian')}
          >
            Cartesian
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="panel">
        <div className="panel__title">Stats</div>
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-card__value">{stats.reward.toFixed(3)}</div>
            <div className="stat-card__label">Reward</div>
          </div>
          <div className="stat-card">
            <div className="stat-card__value">
              {stats.isSuccess ? 'Yes' : 'No'}
            </div>
            <div className="stat-card__label">Success</div>
          </div>
        </div>
      </div>

      {/* Recording */}
      <div className="panel">
        <div className="panel__title">Recording</div>
        {stats.recording && (
          <div className="recording-indicator">
            <span className="recording-indicator__dot" />
            Recording
          </div>
        )}
        <div className="btn-group" style={{ marginTop: 8 }}>
          <button
            className="btn btn--danger btn--sm"
            onClick={() => sendCommand({ type: 'start_recording' })}
            disabled={!isConnected || stats.recording}
          >
            Record
          </button>
          <button
            className="btn btn--ghost btn--sm"
            onClick={() => sendCommand({ type: 'stop_recording' })}
            disabled={!isConnected || !stats.recording}
          >
            Stop
          </button>
        </div>
      </div>
    </div>
  )
}
