import { useEffect } from 'react'

interface VideoStreamProps {
  videoRef: React.RefObject<HTMLVideoElement>
  isConnected: boolean
}

export default function VideoStream({ videoRef, isConnected }: VideoStreamProps) {
  useEffect(() => {
    const el = videoRef.current
    if (el) {
      el.play().catch(() => {
        el.muted = true
        el.play().catch(() => {})
      })
    }
  }, [videoRef, isConnected])

  return (
    <div className="video-fullscreen">
      <video ref={videoRef} autoPlay playsInline muted />
    </div>
  )
}
