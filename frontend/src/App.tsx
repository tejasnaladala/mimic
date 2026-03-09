import React, { useState, useEffect, useCallback } from 'react'
import VideoStream from './components/VideoStream'
import Controls from './components/Controls'
import TouchJoystick from './components/TouchJoystick'
import EpisodeManager from './components/EpisodeManager'
import { useWebRTC } from './hooks/useWebRTC'
import { useGamepad } from './hooks/useGamepad'

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

const SHORTCUTS = [
  { key: 'W/S', desc: 'Joint 0' },
  { key: 'A/D', desc: 'Joint 1' },
  { key: 'Q/E', desc: 'Joint 2' },
  { key: 'R/F', desc: 'Joint 3' },
  { key: 'T/G', desc: 'Joint 4' },
  { key: 'Y/H', desc: 'Joint 5' },
  { key: 'U/J', desc: 'Joint 6' },
  { key: 'O/L', desc: 'Gripper' },
  { key: 'Space', desc: 'Reset' },
  { key: 'M', desc: 'Toggle Mode' },
]

export default function App() {
  const [envInfo, setEnvInfo] = useState<EnvInfo | null>(null)
  const [controlMode, setControlMode] = useState('joint')
  const [episodeCount, setEpisodeCount] = useState(0)
  const [stats, setStats] = useState<Stats>({
    reward: 0,
    done: false,
    isSuccess: false,
    recording: false,
  })

  const handleStateUpdate = useCallback((state: Record<string, unknown>) => {
    if (state.reward !== undefined) {
      setStats({
        reward: state.reward as number,
        done: (state.done as boolean) ?? false,
        isSuccess: (state.is_success as boolean) ?? false,
        recording: (state.recording as boolean) ?? false,
      })
    }
    if (state.mode !== undefined) {
      setControlMode(state.mode as string)
    }
  }, [])

  const { connect, disconnect, sendCommand, videoRef, isConnected } =
    useWebRTC(handleStateUpdate)

  const { gamepadConnected, gamepadName } = useGamepad(sendCommand, isConnected)

  // Fetch environment info
  useEffect(() => {
    fetch('/api/env-info')
      .then((r) => r.json())
      .then(setEnvInfo)
      .catch(() => {})
  }, [])

  // Keyboard controls
  useEffect(() => {
    const keyMap: Record<string, Record<string, unknown>> = {
      w: { type: 'joint_delta', joint: 0, delta: 1.0 },
      s: { type: 'joint_delta', joint: 0, delta: -1.0 },
      a: { type: 'joint_delta', joint: 1, delta: 1.0 },
      d: { type: 'joint_delta', joint: 1, delta: -1.0 },
      q: { type: 'joint_delta', joint: 2, delta: 1.0 },
      e: { type: 'joint_delta', joint: 2, delta: -1.0 },
      r: { type: 'joint_delta', joint: 3, delta: 1.0 },
      f: { type: 'joint_delta', joint: 3, delta: -1.0 },
      t: { type: 'joint_delta', joint: 4, delta: 1.0 },
      g: { type: 'joint_delta', joint: 4, delta: -1.0 },
      y: { type: 'joint_delta', joint: 5, delta: 1.0 },
      h: { type: 'joint_delta', joint: 5, delta: -1.0 },
      u: { type: 'joint_delta', joint: 6, delta: 1.0 },
      j: { type: 'joint_delta', joint: 6, delta: -1.0 },
      o: { type: 'gripper', value: 1.0 },
      l: { type: 'gripper', value: 0.0 },
    }

    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isConnected) return

      // Ignore when typing in input fields
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return
      }

      if (e.key === ' ') {
        e.preventDefault()
        sendCommand({ type: 'reset' })
        return
      }

      if (e.key.toLowerCase() === 'm') {
        const newMode = controlMode === 'joint' ? 'cartesian' : 'joint'
        setControlMode(newMode)
        sendCommand({ type: 'set_mode', mode: newMode })
        return
      }

      const cmd = keyMap[e.key.toLowerCase()]
      if (cmd) {
        sendCommand(cmd)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isConnected, sendCommand, controlMode])

  // Touch joystick handlers
  const handleLeftJoystick = useCallback(
    (x: number, y: number) => {
      sendCommand({ type: 'cartesian_delta', dx: x, dy: -y, dz: 0 })
    },
    [sendCommand]
  )

  const handleRightJoystick = useCallback(
    (x: number, y: number) => {
      sendCommand({ type: 'cartesian_delta', dx: 0, dy: 0, dz: -y, rz: x })
    },
    [sendCommand]
  )

  const handleJoystickRelease = useCallback(() => {
    // No-op: stop sending deltas
  }, [])

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header__logo">MIMIC</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {gamepadConnected && (
            <span className="badge badge--connected">
              <span className="badge__dot" />
              Gamepad
            </span>
          )}
          <span
            className={`badge ${
              isConnected ? 'badge--connected' : 'badge--disconnected'
            }`}
          >
            <span className="badge__dot" />
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </header>

      {/* Main Content */}
      <div className="main-content">
        <VideoStream videoRef={videoRef} isConnected={isConnected} />
        <div className="sidebar">
          <Controls
            isConnected={isConnected}
            onConnect={connect}
            onDisconnect={disconnect}
            sendCommand={sendCommand}
            envInfo={envInfo}
            controlMode={controlMode}
            onModeChange={setControlMode}
            stats={stats}
          />

          <EpisodeManager
            isConnected={isConnected}
            recording={stats.recording}
            episodeCount={episodeCount}
            sendCommand={sendCommand}
          />

          {/* Touch Joysticks (visible on touch devices) */}
          <div className="panel">
            <div className="panel__title">Touch Controls</div>
            <div className="joystick-container">
              <TouchJoystick
                label="Move XY"
                onMove={handleLeftJoystick}
                onRelease={handleJoystickRelease}
              />
              <TouchJoystick
                label="Z / Rotate"
                onMove={handleRightJoystick}
                onRelease={handleJoystickRelease}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Footer with keyboard shortcuts */}
      <footer className="footer">
        <div className="shortcuts">
          {SHORTCUTS.map((s) => (
            <span key={s.key} className="shortcut">
              <span className="shortcut__key">{s.key}</span>
              {s.desc}
            </span>
          ))}
        </div>
      </footer>
    </div>
  )
}
