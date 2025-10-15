/**
 * Unit tests for EquipmentCard component
 * @vitest-environment jsdom
 */
import { describe, it, expect, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import EquipmentCard from '@/components/EquipmentCard.vue'

describe('EquipmentCard.vue', () => {
  const mockEquipment = {
    id: 1,
    name: 'Hydraulic Pump A',
    equipment_type: 'pump',
    manufacturer: 'HydroTech',
    model: 'HT-3000',
    serial_number: 'SN123456',
    status: 'operational'
  }

  it('renders equipment name correctly', () => {
    const wrapper = mount(EquipmentCard, {
      props: { equipment: mockEquipment }
    })
    expect(wrapper.text()).toContain('Hydraulic Pump A')
  })

  it('displays equipment type and model', () => {
    const wrapper = mount(EquipmentCard, {
      props: { equipment: mockEquipment }
    })
    expect(wrapper.text()).toContain('pump')
    expect(wrapper.text()).toContain('HT-3000')
  })

  it('shows serial number', () => {
    const wrapper = mount(EquipmentCard, {
      props: { equipment: mockEquipment }
    })
    expect(wrapper.text()).toContain('SN123456')
  })

  it('emits view-details event when clicked', async () => {
    const wrapper = mount(EquipmentCard, {
      props: { equipment: mockEquipment }
    })
    
    await wrapper.find('.equipment-card').trigger('click')
    expect(wrapper.emitted('view-details')).toBeTruthy()
    expect(wrapper.emitted('view-details')[0]).toEqual([mockEquipment.id])
  })

  it('applies correct status class', () => {
    const wrapper = mount(EquipmentCard, {
      props: { equipment: mockEquipment }
    })
    expect(wrapper.find('.status-operational').exists()).toBe(true)
  })

  it('renders with minimal required props', () => {
    const minimalEquipment = {
      id: 2,
      name: 'Test Equipment',
      equipment_type: 'motor'
    }
    const wrapper = mount(EquipmentCard, {
      props: { equipment: minimalEquipment }
    })
    expect(wrapper.text()).toContain('Test Equipment')
  })
})
