import { useState, useEffect } from 'react'
import { getMe, logout } from './api'
import Layout from './components/Layout'
import LoginPage from './components/LoginPage'
import Welcome from './components/Welcome'
import ChatPanel from './components/ChatPanel'
import SensorDashboard from './components/SensorDashboard'
import SensorChart from './components/SensorChart'
import AlertFeed from './components/AlertFeed'
import CoopLog from './components/CoopLog'
import AutomationPanel from './components/AutomationPanel'
import Weather from './components/Weather'
import MyChickens from './components/MyChickens'
import HeatmapView from './components/HeatmapView'
import Packages from './components/Packages'
import ResponseReview from './components/ResponseReview'

const TABS = [
  { id: 'welcome',    label: 'Home',        title: 'ChickenCoopComfort' },
  { id: 'chat',       label: 'Chat',        title: 'Chat — ChickenCoopComfort' },
  { separator: true, label: 'live' },
  { id: 'sensors',    label: 'Sensors',     title: 'Sensors — ChickenCoopComfort' },
  { id: 'charts',     label: 'Charts',      title: 'Charts — ChickenCoopComfort' },
  { id: 'alerts',     label: 'Alerts',      title: 'Alerts — ChickenCoopComfort' },
  { id: 'weather',    label: 'Weather',     title: 'Weather — ChickenCoopComfort' },
  { separator: true, label: 'coop' },
  { id: 'eggs',       label: 'Coop Log',    title: 'Coop Log — ChickenCoopComfort' },
  { id: 'automation', label: 'Automation',  title: 'Automation — ChickenCoopComfort' },
  { separator: true, label: 'flock' },
  { id: 'chickens',   label: 'My Chickens', title: 'My Chickens — ChickenCoopComfort' },
  { id: 'flock',      label: 'Flock',       title: 'Flock — ChickenCoopComfort' },
  { separator: true, label: 'tools' },
  { id: 'review',     label: 'Review',      title: 'Review — ChickenCoopComfort' },
  { id: 'packages',   label: 'Packages',    title: 'Packages — ChickenCoopComfort' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('welcome')
  const [currentUser, setCurrentUser] = useState(null)
  const [authChecked, setAuthChecked] = useState(false)

  useEffect(() => {
    getMe().then(user => {
      setCurrentUser(user)
      setAuthChecked(true)
    })
  }, [])

  useEffect(() => {
    const tab = TABS.find(t => t.id === activeTab)
    document.title = tab?.title ?? 'ChickenCoopComfort'
  }, [activeTab])

  function handleLogin(username) {
    setCurrentUser({ username })
  }

  function handleLogout() {
    logout()
    setCurrentUser(null)
  }

  if (!authChecked) return null

  if (!currentUser) {
    return <LoginPage onLogin={handleLogin} />
  }

  return (
    <Layout tabs={TABS} activeTab={activeTab} onTabChange={setActiveTab} currentUser={currentUser} onLogout={handleLogout}>
      <div className={activeTab === 'welcome'    ? 'h-full' : 'hidden'}><Welcome onNavigate={setActiveTab} /></div>
      <div className={activeTab === 'chat'       ? 'h-full' : 'hidden'}><ChatPanel /></div>
      <div className={activeTab === 'sensors'    ? 'h-full' : 'hidden'}><SensorDashboard /></div>
      <div className={activeTab === 'charts'     ? 'h-full' : 'hidden'}>
        <div className="h-full overflow-y-auto px-6 py-8 bg-stone-50 dark:bg-stone-900">
          <div className="max-w-4xl mx-auto animate-fade-in">
            <SensorChart />
          </div>
        </div>
      </div>
      <div className={activeTab === 'alerts'     ? 'h-full' : 'hidden'}><AlertFeed /></div>
      <div className={activeTab === 'review'     ? 'h-full' : 'hidden'}><ResponseReview /></div>
      <div className={activeTab === 'eggs'       ? 'h-full' : 'hidden'}><CoopLog /></div>
      <div className={activeTab === 'automation' ? 'h-full' : 'hidden'}><AutomationPanel /></div>
      <div className={activeTab === 'weather'    ? 'h-full' : 'hidden'}><Weather /></div>
      <div className={activeTab === 'chickens'   ? 'h-full' : 'hidden'}><MyChickens currentUser={currentUser} /></div>
      <div className={activeTab === 'packages'   ? 'h-full' : 'hidden'}><Packages /></div>
      <div className={activeTab === 'flock'      ? 'h-full' : 'hidden'}><HeatmapView /></div>
    </Layout>
  )
}
