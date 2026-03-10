interface ControlsProps {
  isConnected: boolean
  onConnect: () => void
  onDisconnect: () => void
  sendCommand: (cmd: Record<string, unknown>) => void
  controlMode: string
  onModeChange: (mode: string) => void
}

export default function Controls({
  isConnected,
  onConnect,
  onDisconnect,
  sendCommand,
  controlMode,
  onModeChange,
}: ControlsProps) {
  const handleModeToggle = () => {
    const newMode = controlMode === 'joint' ? 'cartesian' : 'joint'
    onModeChange(newMode)
    sendCommand({ type: 'set_mode', mode: newMode })
  }

  return (
    <div className="hud-controls">
      <div className="hud-btn-group">
        {!isConnected ? (
          <button className="hud-btn" onClick={onConnect}>
            [ CONNECT ]
          </button>
        ) : (
          <button className="hud-btn hud-btn--danger" onClick={onDisconnect}>
            [ DISCONNECT ]
          </button>
        )}
        <button
          className="hud-btn"
          onClick={() => sendCommand({ type: 'reset' })}
          disabled={!isConnected}
        >
          [ RESET ]
        </button>
      </div>
      <button
        className="hud-btn hud-btn--secondary"
        onClick={handleModeToggle}
        disabled={!isConnected}
      >
        MODE: {controlMode}
      </button>
    </div>
  )
}
