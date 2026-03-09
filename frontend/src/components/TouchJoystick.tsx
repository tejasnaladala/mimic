import React, { useRef, useState, useCallback, useEffect } from 'react'

interface TouchJoystickProps {
  label: string
  onMove: (x: number, y: number) => void
  onRelease: () => void
}

export default function TouchJoystick({ label, onMove, onRelease }: TouchJoystickProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [knobPos, setKnobPos] = useState({ x: 0, y: 0 })
  const activeTouch = useRef<number | null>(null)

  const getRelativePos = useCallback(
    (clientX: number, clientY: number) => {
      const el = containerRef.current
      if (!el) return { x: 0, y: 0 }
      const rect = el.getBoundingClientRect()
      const cx = rect.left + rect.width / 2
      const cy = rect.top + rect.height / 2
      const radius = rect.width / 2
      let dx = (clientX - cx) / radius
      let dy = (clientY - cy) / radius
      // Clamp to unit circle
      const mag = Math.sqrt(dx * dx + dy * dy)
      if (mag > 1) {
        dx /= mag
        dy /= mag
      }
      return { x: dx, y: dy }
    },
    []
  )

  const handleTouchStart = useCallback(
    (e: React.TouchEvent) => {
      if (activeTouch.current !== null) return
      const touch = e.changedTouches[0]
      activeTouch.current = touch.identifier
      const pos = getRelativePos(touch.clientX, touch.clientY)
      setKnobPos(pos)
      onMove(pos.x, pos.y)
    },
    [getRelativePos, onMove]
  )

  const handleTouchMove = useCallback(
    (e: React.TouchEvent) => {
      for (let i = 0; i < e.changedTouches.length; i++) {
        const touch = e.changedTouches[i]
        if (touch.identifier === activeTouch.current) {
          const pos = getRelativePos(touch.clientX, touch.clientY)
          setKnobPos(pos)
          onMove(pos.x, pos.y)
          break
        }
      }
    },
    [getRelativePos, onMove]
  )

  const handleTouchEnd = useCallback(
    (e: React.TouchEvent) => {
      for (let i = 0; i < e.changedTouches.length; i++) {
        if (e.changedTouches[i].identifier === activeTouch.current) {
          activeTouch.current = null
          setKnobPos({ x: 0, y: 0 })
          onRelease()
          break
        }
      }
    },
    [onRelease]
  )

  // Clean up on unmount
  useEffect(() => {
    return () => {
      activeTouch.current = null
    }
  }, [])

  const knobStyle: React.CSSProperties = {
    transform: `translate(calc(-50% + ${knobPos.x * 34}px), calc(-50% + ${knobPos.y * 34}px))`,
  }

  return (
    <div>
      <div
        ref={containerRef}
        className="joystick"
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
        onTouchCancel={handleTouchEnd}
      >
        <div className="joystick__knob" style={knobStyle} />
      </div>
      <div className="joystick__label">{label}</div>
    </div>
  )
}
