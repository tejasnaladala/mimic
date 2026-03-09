import { useRef, useState, useEffect, useCallback } from 'react'

interface UseGamepadReturn {
  gamepadConnected: boolean
  gamepadName: string | null
}

/** Deadzone threshold for analog sticks. */
const DEADZONE = 0.15

function applyDeadzone(value: number): number {
  return Math.abs(value) > DEADZONE ? value : 0
}

export function useGamepad(
  sendCommand: (command: Record<string, unknown>) => void,
  isConnected: boolean
): UseGamepadReturn {
  const [gamepadConnected, setGamepadConnected] = useState(false)
  const [gamepadName, setGamepadName] = useState<string | null>(null)
  const rafRef = useRef<number>(0)
  const prevButtonsRef = useRef<boolean[]>([])

  const poll = useCallback(() => {
    const gamepads = navigator.getGamepads()
    let gp: Gamepad | null = null

    for (const pad of gamepads) {
      if (pad && pad.connected) {
        gp = pad
        break
      }
    }

    if (!gp) {
      if (gamepadConnected) {
        setGamepadConnected(false)
        setGamepadName(null)
      }
      rafRef.current = requestAnimationFrame(poll)
      return
    }

    if (!gamepadConnected) {
      setGamepadConnected(true)
      setGamepadName(gp.id)
    }

    // Only send commands when WebRTC is connected
    if (isConnected) {
      // Axes: left stick (0,1) = XY, right stick (2,3) = Z + rotation
      const lx = applyDeadzone(gp.axes[0] ?? 0)
      const ly = applyDeadzone(gp.axes[1] ?? 0)
      const rx = applyDeadzone(gp.axes[2] ?? 0)
      const ry = applyDeadzone(gp.axes[3] ?? 0)

      if (lx !== 0 || ly !== 0 || rx !== 0 || ry !== 0) {
        sendCommand({
          type: 'cartesian_delta',
          dx: lx,
          dy: -ly, // Invert Y for intuitive up = forward
          dz: -ry, // Right stick Y for Z
          rz: rx,  // Right stick X for rotation
        })
      }

      // Buttons (using standard gamepad mapping)
      const buttons = gp.buttons.map((b) => b.pressed)
      const prev = prevButtonsRef.current

      // A button (0) = gripper close
      if (buttons[0] && !prev[0]) {
        sendCommand({ type: 'gripper', value: 0.0 })
      }
      // B button (1) = gripper open
      if (buttons[1] && !prev[1]) {
        sendCommand({ type: 'gripper', value: 1.0 })
      }
      // X button (2) = reset
      if (buttons[2] && !prev[2]) {
        sendCommand({ type: 'reset' })
      }
      // Y button (3) = toggle mode
      if (buttons[3] && !prev[3]) {
        sendCommand({ type: 'toggle_mode' })
      }

      prevButtonsRef.current = buttons
    }

    rafRef.current = requestAnimationFrame(poll)
  }, [sendCommand, isConnected, gamepadConnected])

  useEffect(() => {
    rafRef.current = requestAnimationFrame(poll)
    return () => {
      cancelAnimationFrame(rafRef.current)
    }
  }, [poll])

  return { gamepadConnected, gamepadName }
}
