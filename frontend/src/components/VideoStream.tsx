import React, { useEffect } from 'react'

interface VideoStreamProps {
  videoRef: React.RefObject<HTMLVideoElement>
  isConnected: boolean
}

export default function VideoStream({ videoRef, isConnected }: VideoStreamProps) {
  useEffect(() => {
    // Attempt autoplay when video element is available
    const el = videoRef.current
    if (el) {
      el.play().catch(() => {
        // Autoplay blocked: mute and try again
        el.muted = true
        el.play().catch(() => {})
      })
    }
  }, [videoRef, isConnected])

  return (
    <div className="video-area">
      <video ref={videoRef} autoPlay playsInline muted />
      {!isConnected && (
        <div className="video-overlay">
          <div className="video-overlay__pulse">
            Click Connect to start teleoperation
          </div>
        </div>
      )}
    </div>
  )
}
