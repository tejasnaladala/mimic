import { useRef, useState, useCallback, useEffect } from 'react'

interface UseWebRTCReturn {
  connect: () => Promise<void>
  disconnect: () => void
  sendCommand: (command: Record<string, unknown>) => void
  videoRef: React.RefObject<HTMLVideoElement>
  isConnected: boolean
  dataChannel: RTCDataChannel | null
}

export function useWebRTC(
  onStateUpdate?: (state: Record<string, unknown>) => void
): UseWebRTCReturn {
  const videoRef = useRef<HTMLVideoElement>(null!)
  const pcRef = useRef<RTCPeerConnection | null>(null)
  const dcRef = useRef<RTCDataChannel | null>(null)
  const [isConnected, setIsConnected] = useState(false)

  const disconnect = useCallback(() => {
    if (dcRef.current) {
      dcRef.current.close()
      dcRef.current = null
    }
    if (pcRef.current) {
      pcRef.current.close()
      pcRef.current = null
    }
    setIsConnected(false)
  }, [])

  const connect = useCallback(async () => {
    // Clean up any existing connection
    disconnect()

    const pc = new RTCPeerConnection({
      iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
    })
    pcRef.current = pc

    // Create data channel for sending commands
    const dc = pc.createDataChannel('commands', { ordered: true })
    dcRef.current = dc

    dc.onopen = () => {
      setIsConnected(true)
    }

    dc.onclose = () => {
      setIsConnected(false)
    }

    dc.onmessage = (event) => {
      try {
        const state = JSON.parse(event.data)
        if (onStateUpdate) {
          onStateUpdate(state)
        }
      } catch {
        // Ignore malformed messages
      }
    }

    // Handle incoming video track
    pc.ontrack = (event) => {
      if (videoRef.current) {
        videoRef.current.srcObject = event.streams[0]
      }
    }

    pc.onconnectionstatechange = () => {
      if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
        setIsConnected(false)
      }
    }

    // Create and send offer
    const offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    // Wait for ICE gathering to complete
    await new Promise<void>((resolve) => {
      if (pc.iceGatheringState === 'complete') {
        resolve()
      } else {
        pc.onicegatheringstatechange = () => {
          if (pc.iceGatheringState === 'complete') {
            resolve()
          }
        }
      }
    })

    // Send offer to server
    const response = await fetch('/api/offer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sdp: pc.localDescription!.sdp,
        type: pc.localDescription!.type,
      }),
    })

    const answer = await response.json()
    await pc.setRemoteDescription(new RTCSessionDescription(answer))
  }, [disconnect, onStateUpdate])

  const sendCommand = useCallback((command: Record<string, unknown>) => {
    if (dcRef.current && dcRef.current.readyState === 'open') {
      dcRef.current.send(JSON.stringify(command))
    }
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [disconnect])

  return {
    connect,
    disconnect,
    sendCommand,
    videoRef,
    isConnected,
    dataChannel: dcRef.current,
  }
}
