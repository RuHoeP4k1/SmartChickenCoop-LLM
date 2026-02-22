import { useState } from 'react'
import Layout from './components/Layout'
import ChatPanel from './components/ChatPanel'
import SensorDashboard from './components/SensorDashboard'
import AlertFeed from './components/AlertFeed'

const TABS = [
  { id: 'chat',    label: 'Ask ChatKippieTee' },
  { id: 'sensors', label: 'Live Sensors' },
  { id: 'alerts',  label: 'Alert Feed' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('chat')

  return (
    <Layout tabs={TABS} activeTab={activeTab} onTabChange={setActiveTab}>
      <div className={activeTab === 'chat'    ? 'h-full' : 'hidden'}><ChatPanel /></div>
      <div className={activeTab === 'sensors' ? 'h-full' : 'hidden'}><SensorDashboard /></div>
      <div className={activeTab === 'alerts'  ? 'h-full' : 'hidden'}><AlertFeed /></div>
    </Layout>
  )
}
