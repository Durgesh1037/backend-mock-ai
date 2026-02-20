import {Sequelize, DataTypes, Model} from 'sequelize';

const userStreaks = (sequelize: Sequelize) => {
  const UserStreaks = sequelize.define('userStreak', {
    uuid: {
      type: Sequelize.UUID,
      defaultValue: Sequelize.UUIDV4,
      allowNull: false,
      primaryKey: true,
    },
    userId: {
      type: Sequelize.INTEGER,
      allowNull: false,
      unique: true,
    },
    streakCount: {
      type: Sequelize.INTEGER,
      allowNull: false,
      defaultValue: 0,
    },
    lastActiveDate: {
      type: Sequelize.DATEONLY,
      allowNull: false,
    },
    streakDays: {
      type: Sequelize.TEXT('long'),
      allowNull: true,
    },
    year: {    
        type: Sequelize.INTEGER,
        allowNull: false,
    },
    month: {
        type: Sequelize.INTEGER,
        allowNull: false,
    },
  });

  return UserStreaks;
};

export default userStreaks;