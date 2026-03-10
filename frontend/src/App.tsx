import { useState, useEffect, useCallback, useRef } from 'react'
import VideoStream from './components/VideoStream'
import Controls from './components/Controls'
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
  { key: 'W/S', desc: 'J0' },
  { key: 'A/D', desc: 'J1' },
  { key: 'Q/E', desc: 'J2' },
  { key: 'R/F', desc: 'J3' },
  { key: 'T/G', desc: 'J4' },
  { key: 'Y/H', desc: 'J5' },
  { key: 'U/J', desc: 'J6' },
  { key: 'O/L', desc: 'Grip' },
  { key: 'Space', desc: 'Reset' },
  { key: 'M', desc: 'Mode' },
]

export default function App() {
  const [envInfo, setEnvInfo] = useState<EnvInfo | null>(null)
  const [controlMode, setControlMode] = useState('joint')
  const [episodeCount, setEpisodeCount] = useState(0)
  const [jointPos, setJointPos] = useState<number[]>([])
  const [fps, setFps] = useState(0)
  const [stats, setStats] = useState<Stats>({
    reward: 0,
    done: false,
    isSuccess: false,
    recording: false,
  })

  // FPS counter
  const frameCount = useRef(0)
  const lastFpsTime = useRef(Date.now())

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
    if (state.joint_pos !== undefined) {
      setJointPos(state.joint_pos as number[])
    }
    if (state.episode_count !== undefined) {
      setEpisodeCount(state.episode_count as number)
    }

    // FPS calculation
    frameCount.current++
    const now = Date.now()
    if (now - lastFpsTime.current >= 1000) {
      setFps(frameCount.current)
      frameCount.current = 0
      lastFpsTime.current = now
    }
  }, [])

  const { connect, disconnect, sendCommand, videoRef, isConnected } =
    useWebRTC(handleStateUpdate)

  const { gamepadConnected } = useGamepad(sendCommand, isConnected)

  // Fetch environment info
  useEffect(() => {
    fetch('/api/env-info')
      .then((r) => r.json())
      .then(setEnvInfo)
      .catch(() => {})
  }, [])

  // Keyboard controls (keydown)
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

    // Map keys to joint indices for keyup release messages
    const keyJointMap: Record<string, number> = {
      w: 0, s: 0,
      a: 1, d: 1,
      q: 2, e: 2,
      r: 3, f: 3,
      t: 4, g: 4,
      y: 5, h: 5,
      u: 6, j: 6,
    }

    const handleKeyDown = (ev: KeyboardEvent) => {
      if (!isConnected) return

      // Ignore when typing in input fields
      if (ev.target instanceof HTMLInputElement || ev.target instanceof HTMLTextAreaElement) {
        return
      }

      if (ev.key === ' ') {
        ev.preventDefault()
        sendCommand({ type: 'reset' })
        return
      }

      if (ev.key.toLowerCase() === 'm') {
        const newMode = controlMode === 'joint' ? 'cartesian' : 'joint'
        setControlMode(newMode)
        sendCommand({ type: 'set_mode', mode: newMode })
        return
      }

      const cmd = keyMap[ev.key.toLowerCase()]
      if (cmd) {
        sendCommand(cmd)
      }
    }

    const handleKeyUp = (ev: KeyboardEvent) => {
      if (!isConnected) return
      if (ev.target instanceof HTMLInputElement || ev.target instanceof HTMLTextAreaElement) {
        return
      }

      const joint = keyJointMap[ev.key.toLowerCase()]
      if (joint !== undefined) {
        sendCommand({ type: 'joint_release', joint })
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('keyup', handleKeyUp)
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
    }
  }, [isConnected, sendCommand, controlMode])

  return (
    <div className="hud-container">
      {/* Full-bleed video background */}
      <VideoStream videoRef={videoRef} isConnected={isConnected} />

      {/* Scanline overlay */}
      <div className="scanline-overlay" />

      {/* TOP-LEFT: Logo + Version */}
      <div className="hud-panel hud-panel--top-left">
        <div className="hud-logo">MIMIC</div>
        <div className="hud-version">
          {envInfo ? envInfo.name : 'TELEOP V0.1'}
        </div>
        {envInfo && (
          <div className="hud-mode">
            {controlMode} // {envInfo.control_hz}HZ
          </div>
        )}
      </div>

      {/* TOP-RIGHT: Connection + FPS */}
      <div className="hud-panel hud-panel--top-right">
        <div className="hud-status">
          <div
            className={`hud-status__dot ${
              !isConnected ? 'hud-status__dot--disconnected' : ''
            }`}
          />
          <span
            className={`hud-status__label ${
              !isConnected ? 'hud-status__label--disconnected' : ''
            }`}
          >
            {isConnected ? 'ONLINE' : 'OFFLINE'}
          </span>
        </div>
        {isConnected && (
          <div className="hud-fps">{fps} FPS</div>
        )}
        {gamepadConnected && (
          <div className="hud-gamepad">
            <div className="hud-gamepad__dot" />
            GAMEPAD
          </div>
        )}
      </div>

      {/* BOTTOM-LEFT: Joint angles readout */}
      <div
        className={`hud-panel hud-panel--bottom-left ${
          !isConnected ? 'hud-panel--disconnected' : ''
        }`}
      >
        <JointReadout jointPos={jointPos} />
      </div>

      {/* BOTTOM-RIGHT: Controls + Episode */}
      <div className="hud-panel hud-panel--bottom-right">
        <Controls
          isConnected={isConnected}
          onConnect={connect}
          onDisconnect={disconnect}
          sendCommand={sendCommand}
          controlMode={controlMode}
          onModeChange={setControlMode}
        />
        <div className="hud-divider" />
        <EpisodeManager
          isConnected={isConnected}
          recording={stats.recording}
          episodeCount={episodeCount}
          sendCommand={sendCommand}
        />
      </div>

      {/* BOTTOM-CENTER: Keyboard shortcuts */}
      <div className="hud-panel hud-panel--bottom-center">
        <div className="hud-shortcuts">
          {SHORTCUTS.map((s) => (
            <span key={s.key} className="hud-shortcut">
              <span className="hud-shortcut__key">{s.key}</span>
              {s.desc}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}

/* ---- Joint Readout Sub-component ---- */

function JointReadout({ jointPos }: { jointPos: number[] }) {
  const prevRef = useRef<number[]>([])
  const [changedJoints, setChangedJoints] = useState<Set<number>>(new Set())

  useEffect(() => {
    const changed = new Set<number>()
    for (let i = 0; i < jointPos.length; i++) {
      if (prevRef.current[i] !== undefined && Math.abs(jointPos[i] - prevRef.current[i]) > 0.001) {
        changed.add(i)
      }
    }
    prevRef.current = [...jointPos]
    setChangedJoints(changed)

    // Clear glow after a short delay
    const timer = setTimeout(() => setChangedJoints(new Set()), 200)
    return () => clearTimeout(timer)
  }, [jointPos])

  const joints = jointPos.length > 0 ? jointPos : [0, 0, 0, 0, 0, 0, 0]

  return (
    <div className="hud-joints">
      <div className="hud-joints__title">JOINT ANGLES</div>
      {joints.map((val, i) => (
        <div key={i} className="hud-joints__row">
          <span className="hud-joints__label">J{i}</span>
          <span
            className={`hud-joints__value ${
              changedJoints.has(i) ? 'hud-joints__value--changed' : ''
            }`}
          >
            {val.toFixed(4)}
          </span>
        </div>
      ))}
    </div>
  )
}
